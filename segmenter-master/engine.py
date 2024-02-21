import numpy as np
import torch
import math

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os, sys

# our attention loss function
def atten_loss_compute(atten_value, seg_gt, atten_loss_head = [0,1,2]):
    b, h, w = seg_gt.shape
    stride = 16
    total_cls = 256
    real_cls = 150
    temputure = 0.2
    head = 3 #3

    cc = torch.eye(real_cls).cuda()
    x4 = cc.repeat(b, 1, 1)


    seg_gt_onehot = F.one_hot(seg_gt, num_classes = total_cls).permute(0, 3, 1, 2).float()
    seg_gt_onehot_pool = F.avg_pool2d(seg_gt_onehot, (stride, stride), stride=stride)  # may be max pool
    seg_gt_onehot_f = seg_gt_onehot_pool.reshape(b, total_cls, int(h/stride) * int(w/stride))
    seg_gt_onehot_re = seg_gt_onehot_f[:, :real_cls, :]
    seg_gt_onehot_re255 = 1 - seg_gt_onehot_f[:, -1, :]
    seg_gt_onehot_re1 = torch.cat([seg_gt_onehot_re, x4], dim=2)
    #seg_gt_onehot_soft = F.softmax(seg_gt_onehot_re1 / temputure, dim = -1) #* seg_gt_onehot_re1 old
    seg_gt_onehot_re_sum = torch.sum(seg_gt_onehot_re, dim=2)
    seg_gt_onehot_re_sumhot = 1 - F.relu(1 - seg_gt_onehot_re_sum)
    seg_gt_onehot_active = torch.cat([seg_gt_onehot_re255, seg_gt_onehot_re_sumhot], dim = 1)
    seg_gt_onehot_active1 = torch.unsqueeze(seg_gt_onehot_active, dim=1)
    seg_gt_onehot_active0 = torch.unsqueeze(seg_gt_onehot_re_sumhot, dim=1)

    seg_gt_onehot_re1 = seg_gt_onehot_re1.permute(0, 2, 1)   # new
    attn = (seg_gt_onehot_re1[:, -real_cls:, :] @ seg_gt_onehot_re1.transpose(-2, -1)) / temputure   # new
    seg_gt_onehot_soft = attn.softmax(dim=-1)   # new


    atten_loss_out = 0
    for idx_a, atten_value_tmp in enumerate(atten_value):
        for idx_c in range(head):
            if idx_c in atten_loss_head:
                atten_loss_out_tmp = - seg_gt_onehot_soft[:, :, :] * torch.log(atten_value_tmp[:, idx_c, :, :])
                total_num_cls = torch.sum(seg_gt_onehot_re255) * torch.sum(seg_gt_onehot_re_sumhot) / b / b
                atten_loss_out_tmp1 = (seg_gt_onehot_active0 @ atten_loss_out_tmp @ (seg_gt_onehot_active1).permute(0, 2, 1)) / total_num_cls
                atten_loss_out += torch.mean(atten_loss_out_tmp1)
    return atten_loss_out

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    atten_loss = False,
    atten_loss_weight = 0,
    atten_loss_layer = [1],
    atten_loss_head = [0,1,2],
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)

        with amp_autocast():
            if atten_loss:
                seg_pred, atten_value = model.forward(im, atten_loss, atten_loss_layer)
                atten_loss_out = atten_loss_compute(atten_value, seg_gt, atten_loss_head) * atten_loss_weight

                loss_bce = criterion(seg_pred, seg_gt)
                loss_value = loss_bce.item()

                loss = loss_bce + atten_loss_out

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value), force=True)

                optimizer.zero_grad()
                if loss_scaler is not None:
                    loss_scaler(
                        loss,
                        optimizer,
                        parameters=model.parameters(),
                    )
                else:
                    loss.backward()
                    optimizer.step()

                num_updates += 1
                lr_scheduler.step_update(num_updates=num_updates)

                torch.cuda.synchronize()

                logger.update(
                    loss=loss_bce.item(),
                    loss_atten=atten_loss_out.item(),
                    loss_sum=loss.item(),
                    learning_rate=optimizer.param_groups[0]["lr"],
                )

            else:

                seg_pred = model.forward(im, atten_loss)
                loss = criterion(seg_pred, seg_gt)
                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value), force=True)

                optimizer.zero_grad()
                if loss_scaler is not None:
                    loss_scaler(
                        loss,
                        optimizer,
                        parameters=model.parameters(),
                    )
                else:
                    loss.backward()
                    optimizer.step()

                num_updates += 1
                lr_scheduler.step_update(num_updates=num_updates)

                torch.cuda.synchronize()

                logger.update(
                    loss=loss.item(),
                    learning_rate=optimizer.param_groups[0]["lr"],
                )

    return logger


class Logger_lmc(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
    log_dir = None,
    epoch = -1
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    val_seg_pred = {}
    model.eval()
    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                ims,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
            )
            seg_pred = seg_pred.argmax(0)

        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename] = seg_pred


    if log_dir is not None:
        max_save = 100
        now_save = 0
        save_dir = log_dir + '/' + str(epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for k, v in val_seg_pred.items():
            if now_save > max_save:
                break
            now_save += 1
            plt.imsave(save_dir + '/' + k, v)


    val_seg_pred = gather_data(val_seg_pred)
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        data_loader.unwrapped.n_cls,
        ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger
