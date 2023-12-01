import numpy as np
import torch
from net.tdcn import TDCN
import torch.optim as optim
from net.data_loader import gt2attengt, Signle_Loader
from torch.utils.data import DataLoader
from os.path import join, isdir
from utils import Logger, save_checkpoint, Averagvalue
import sys
import os
import torchvision
import time
import random
import shutil
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        # m.weight.data.normal_(0, 0.01)
        if not m.weight.requires_grad:
            return

        if m.weight.data.shape == torch.Size([1, 4, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.25)
        else:
            m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

    if isinstance(m, nn.Linear):
        if not m.weight.requires_grad:
            return
        m.weight.data.normal_(0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model():
    nest = TDCN(
        templates=[4, 4, 2, 1],
        #dim=[42, 84, 168, 168],
        dim=[6, 12, 24, 24],
        heads=[2, 4, 8, 8],
        window_size=[1, 2, 4, 4],
        cnn_repeats=(1, 1, 1, 1),
        block_repeats=(1, 1, 1, 1),
        stride=[2, 2, 2],
        pos_embbing=[False, True, True, True],
        cls=[False, True, True, True],
        mlp_mult=0.3,
        bn=True,
        head_mode='cat',  # 'cat' 'plus'
        bias=True,
        pos_embbding_mode='relative',  # relative absolute
        trans_activate=nn.GELU(),
        cnn_activate=nn.GELU(),
        fea_dim_head=16,
        cnn_bn=True,
        cnn_bias=True,
    )
    return nest

def softsoftmax_loss(pred, label):
    cost = - label * torch.log(pred)
    return torch.mean(cost)


def cross_entropy_loss_RCF_adboost2bi_hm(prediction, label, weight_last, level, Shrinkage, mask_now):
    _, _, w, h = label.shape

    label = label.long()
    if level == 0:
        mask = label.float().clone()
        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        mask[mask == 2] = 0
        mask = mask * mask_now
        weight_last = mask.clone()
    else:
        mask = weight_last.clone()

    cost = torch.nn.functional.binary_cross_entropy(prediction.float(), label.float(), weight=mask,
                                                    reduce=False)
    cost_total = torch.sum(cost)
    pred_now = prediction.detach()

    pred_now = torch.clamp(pred_now, min=1 / 1000, max=999 / 1000)
    gm_now = 0.5 * torch.log(pred_now / (1 - pred_now))

    label_mask = label.float().clone()
    label_mask[label_mask == 1] = 1
    label_mask[label_mask == 0] = -1
    label_mask[label_mask == 2] = 0

    weight_new = weight_last * torch.exp(- label_mask * gm_now * Shrinkage)
    if torch.sum(weight_last) == 0:
        weight_new = weight_new
    else:
        weight_new = weight_new / torch.sum(weight_new) * torch.sum(weight_last)

    # mask_n2p = mask.cpu().detach().numpy()[0,0]
    # mask_np1 = mask_cp.cpu().detach().numpy()[0,0]
    # last_np = weight_last.cpu().detach().numpy()[0,0]
    # new_np = weight_new.cpu().detach().numpy()[0,0]

    return cost_total, weight_new


def test(model, test_loader, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for idx, (image, filename) in enumerate(test_loader):
        image = image.cuda()
        _, _, H, W = image.shape
        results, _, _ = model(image)
        results_cpu = []
        for i in results:
            results_cpu.append(i.detach().cpu())
        results_all = torch.zeros((len(results_cpu), 1, H, W))
        for i in range(len(results_cpu)):
            results_all[i, 0, :, :] = results_cpu[i]
        torchvision.utils.save_image(results_all, join(save_dir, "%s.jpg" % filename))
        torchvision.utils.save_image(results_all[-1], join(save_dir, "%s.png" % filename))


def train(train_loader, model, optimizer, epoch, itersize, print_freq, maxepoch,
          save_dir, Shrinkage=1.0, aloss_weight=1, gt2attengt_class=None):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    losses_bce = Averagvalue()
    losses_att = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()
    epoch_loss = []
    counter = 0
    optimizer.zero_grad()
    for i, (image, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()
        gt_atten = gt2attengt_class.build_attention_gt(label)

        outputs, _, attens_out = model(image)
        lossbce = torch.zeros(1).cuda()
        lossatten = torch.zeros(1).cuda()

        gt = label
        pred = outputs

        _, _, gth, gtw = gt.shape
        for idss, ij in enumerate(attens_out):
            num_atten = len(ij)
            for paidss, paij in enumerate(ij):
                lossatten += aloss_weight * gth * gtw * softsoftmax_loss(paij, gt_atten[idss][paidss]) / num_atten

        mask_random = torch.rand(label.shape).cuda()
        mask = torch.ones(label.shape).cuda()
        mask_inverse = torch.ones(label.shape).cuda()
        mask[mask_random < 0.5] = 0
        mask_inverse[mask_random > 0.5] = 0

        last_prediction = None
        for idss, ij in enumerate(pred[:-1]):
            loss_bce_each, last_prediction = cross_entropy_loss_RCF_adboost2bi_hm(ij, gt, last_prediction, idss,
                                                                                  Shrinkage, mask)
            lossbce += loss_bce_each * 2

        last_prediction = None
        for idss, ij in enumerate(pred[-2::-1]):
            loss_bce_each, last_prediction = cross_entropy_loss_RCF_adboost2bi_hm(ij, gt, last_prediction, idss,
                                                                                  Shrinkage, mask_inverse)
            lossbce += loss_bce_each * 2

        loss_total = lossbce + lossatten
        counter += 1
        loss = loss_total / itersize
        loss.backward()
        if counter == itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        losses_bce.update((lossbce / itersize).item(), image.size(0))
        losses_att.update((lossatten / itersize).item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if i % print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   'Loss_bce {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=losses_bce) + \
                   'Loss_att {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=losses_att) + \
                   'Lr {lr} (base lr:{lrb})'.format(lr=optimizer.param_groups[0]['lr'],
                                                    lrb=optimizer.defaults['lr'])
            print(info)
            label_out = torch.eq(label, 1).float()
            outputs.append(label_out)

            _, _, H, W = outputs[0].shape
            all_results = torch.zeros((len(outputs), 1, H, W))
            for j in range(len(outputs)):
                all_results[j, 0, :, :] = outputs[j][0, 0, :, :]
            torchvision.utils.save_image(all_results, join(save_dir, "iter-%d.jpg" % i), nrow=len(outputs))

    return losses.avg, epoch_loss

# for BSDS
def main():
    # hyper-parameters
    aloss_weight = 20
    Shrinkage = 0.4

    # dirs
    TMP_DIR = 'lmc'
    dataset_dir = 'J:/HED-PASCAL'

    # training details
    lr = 2e-3
    batch_size = 1
    itersize = 10  # 10
    stepsize = [6]
    gamma = 0.1
    split_value = 76.8  # 76.8 ## 127.5 / 87.5 / 76.8
    maxepoch = 8
    scheduler_bool = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # save&log
    print_freq = 100
    save_freq = 1
    eval_freq = 1
    seed = 1

    seed_torch(seed)
    train_dataset = Signle_Loader(root=dataset_dir, split="train", split_value=split_value)  # 127.5 / 87.5
    test_dataset = Signle_Loader(root=dataset_dir, split="test", split_value=split_value)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, drop_last=True, shuffle=False)

    model = build_model()
    model = model.cuda()
    model.apply(weights_init)

    gt2attengt_class = gt2attengt(model.window_size, model.heads, model.templates, model.cls)

    num_para = count_parameters(model)
    print(num_para)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr, )
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=stepsize, gamma=gamma)

    # log
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    shutil.copy(sys.argv[0], TMP_DIR + '/' + 'config')
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' % ('adam', lr)))
    sys.stdout = log

    train_loss = []
    train_loss_detail = []
    for epoch in range(1, maxepoch):
        print('Current lr_rate: ' + str(optimizer.param_groups[0]['lr']))

        tr_avg_loss, tr_detail_loss = train(
            train_loader, model, optimizer, epoch, itersize, print_freq, maxepoch,
            save_dir=join(TMP_DIR, 'epoch-%d-training-record' % epoch), Shrinkage=Shrinkage,
            aloss_weight=aloss_weight, gt2attengt_class=gt2attengt_class)
        if scheduler_bool:
            scheduler.step()

        # Save checkpoint
        save_file = os.path.join(TMP_DIR, 'checkpoint_epoch{}.pth'.format(epoch))
        if epoch % save_freq == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), }, filename=save_file)

        if epoch % eval_freq == 0:
            test(model, test_loader,
                 save_dir=join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
        log.flush()  # write log

        # save train/val loss/accuracy, save every epoch in case of early stop
        train_loss.append(tr_avg_loss)
        train_loss_detail += tr_detail_loss


if __name__ == '__main__':
    main()
