from torch.utils import data
from os.path import join
from PIL import Image
import numpy as np
import cv2
import torch
from einops import rearrange
import torch.nn as nn

def prepare_image_PIL(im):
    im = im[:, :, ::-1] - np.zeros_like(im)  # rgb to bgr
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im

def prepare_image_cv2(im):
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im

# for attention loss
class gt2attengt():

    def build_convweight(self, weight_np):
        weight_torch = torch.from_numpy(weight_np).unsqueeze(0).unsqueeze(0).float().cuda()
        return weight_torch

    def build_maxpooling(self, gt_now):
        return torch.max_pool2d(gt_now, kernel_size=2, stride=2, ceil_mode=True)

    def __init__(self, window_size, heads, templates, cls, noone = True, clsignore = False, clsattenignore = True,
                 neg_val = -0.5, pos_val = 1 + 1/3, nor_val = -3 / 8, cnor_val = 4, tempt = 1, neg2posweight = 1):
        self.window_size = window_size
        self.heads = heads
        self.templates = templates
        self.cls = cls
        self.noone = noone
        self.tempt = tempt
        self.clsignore = clsignore
        self.clsattenignore = clsattenignore
        self.neg2posweight = neg2posweight
        self.w0 = self.build_convweight(
            np.array([nor_val, nor_val, nor_val, nor_val, cnor_val, nor_val, nor_val, nor_val, nor_val]).reshape(
                [3, 3]))
        self.w1 = self.build_convweight(
            np.array([neg_val, pos_val, neg_val, neg_val, pos_val, neg_val, neg_val, pos_val, neg_val]).reshape([3, 3]))
        self.w2 = self.build_convweight(
            np.array([neg_val, neg_val, neg_val, pos_val, pos_val, pos_val, neg_val, neg_val, neg_val]).reshape([3, 3]))
        self.w3 = self.build_convweight(
            np.array([pos_val, neg_val, neg_val, neg_val, pos_val, neg_val, neg_val, neg_val, pos_val]).reshape([3, 3]))
        self.w4 = self.build_convweight(
            np.array([neg_val, neg_val, pos_val, neg_val, pos_val, neg_val, pos_val, neg_val, neg_val]).reshape([3, 3]))
        self.w5 = self.build_convweight(
            np.array([neg_val, pos_val, neg_val, pos_val, pos_val, neg_val, neg_val, neg_val, neg_val]).reshape([3, 3]))
        self.w6 = self.build_convweight(
            np.array([neg_val, pos_val, neg_val, neg_val, pos_val, pos_val, neg_val, neg_val, neg_val]).reshape([3, 3]))
        self.w7 = self.build_convweight(
            np.array([neg_val, neg_val, neg_val, pos_val, pos_val, neg_val, neg_val, pos_val, neg_val]).reshape([3, 3]))
        self.w8 = self.build_convweight(
            np.array([neg_val, neg_val, neg_val, neg_val, pos_val, pos_val, neg_val, pos_val, neg_val]).reshape([3, 3]))
    def build_attention_gt(self, gt):

        gt = gt.clone()
        gt[gt == 0] = -1
        gt[gt == 2] = 0
        gt[gt == 1] = 1

        gt_attens = []

        for idx, (window_i, head_i, templates_i, cls_i) in enumerate(zip(self.window_size, self.heads, self.templates, self.cls)):
            #gt_rr2 = gt.clone().cpu().numpy()[0,0]

            gt_fuz = torch.ones(gt.shape).cuda()
            gt_fuz[gt == 0] = 0

            gt_atten = []
            _, _, h, w = gt.shape
            h_block = int(np.ceil(h / window_i) * window_i)
            w_block = int(np.ceil(w / window_i) * window_i)
            if h_block == h and w_block == w:
                gt_tmp = gt.clone()
                gt_fuz_tmp = gt_fuz.clone()
            else:
                gt_tmp = nn.ReflectionPad2d((w_block - w, w_block - w, h_block - h, h_block - h))(gt).clone()
                gt_fuz_tmp = nn.ReflectionPad2d((w_block - w, w_block - w, h_block - h, h_block - h))(gt_fuz).clone()

            num_positive = torch.sum((gt_tmp == 1).float()).float()
            num_negative = torch.sum((gt_tmp == -1).float()).float()
            mask = gt_tmp.clone()
            mask[mask == -1] = self.neg2posweight * num_positive / (num_positive + num_negative)
            mask[mask == 1] = num_negative / (num_positive + num_negative)
            mask[gt_fuz_tmp == 0] = 0

            #mask_nn = mask.detach().cpu().numpy()[0,0]
            #gt_tmpnn = gt_tmp.detach().cpu().numpy()[0,0]
            #gt_fuz_tmpnn = gt_fuz_tmp.detach().cpu().numpy()[0,0]

            # sdsv = torch.sum(mask)
            if cls_i and self.clsignore:
                mask = torch.zeros(gt_tmp.shape).cuda()

            if templates_i == 1 or not self.noone:
            #if templates_i > 0:
                c_merge = torch.conv2d(gt_tmp, weight=self.w0, stride=1, padding=1, groups=1)
            if templates_i > 1:
                c1 = torch.conv2d(gt_tmp, weight=self.w1, stride=1, padding=1, groups=1)
                c2 = torch.conv2d(gt_tmp, weight=self.w2, stride=1, padding=1, groups=1)
                if self.noone:
                    c_merge = torch.cat([c1, c2], dim=1)
                else:
                    c_merge = torch.cat([c_merge, c1, c2], dim=1)
            if templates_i > 2:
                c3 = torch.conv2d(gt_tmp, weight=self.w3, stride=1, padding=1, groups=1)
                c4 = torch.conv2d(gt_tmp, weight=self.w4, stride=1, padding=1, groups=1)
                c_merge = torch.cat([c_merge, c3, c4], dim=1)
            if templates_i > 4:
                c5 = torch.conv2d(gt_tmp, weight=self.w5, stride=1, padding=1, groups=1)
                c6 = torch.conv2d(gt_tmp, weight=self.w6, stride=1, padding=1, groups=1)
                c7 = torch.conv2d(gt_tmp, weight=self.w7, stride=1, padding=1, groups=1)
                c8 = torch.conv2d(gt_tmp, weight=self.w8, stride=1, padding=1, groups=1)
                c_merge = torch.cat([c_merge, c5, c6, c7, c8], dim=1)
            if h_block == h and w_block == w:
                mask_w0 = rearrange(mask, 'b c (b1 h) (b2 w) -> (b b1 b2) (c h w) 1', h=window_i, w=window_i)
                gt_w0 = rearrange(gt_tmp, 'b c (b1 h) (b2 w) -> (b b1 b2) (c h w) 1', h=window_i, w=window_i)
                c_merge_w0 = rearrange(c_merge, 'b c (b1 h) (b2 w) -> (b b1 b2) (c h w) 1', h=window_i, w=window_i)
                gt_c0 = (gt_w0 @ c_merge_w0.transpose(-2, -1)) / self.tempt
                gt_atten0 = torch.nn.functional.softmax(gt_c0, dim=-1) * mask_w0
                if cls_i and not self.clsignore and not self.clsattenignore:
                    hw, win2, temwin2 = gt_atten0.shape
                    avr_va = 1 / (temwin2 + 1)
                    mask_w0_cls = torch.sum(gt_atten0, dim=-1, keepdim=True)
                    gt_atten0 = gt_atten0 * temwin2 / (temwin2 + 1)
                    cls_atten_d2 = avr_va * mask_w0_cls * torch.ones([hw, win2, 1]).cuda()
                    cls_atten_d1 = avr_va * self.neg2posweight * num_positive / (num_positive + num_negative) * torch.ones([hw, 1, temwin2 + 1]).cuda()
                    gt_atten0 = torch.cat([gt_atten0, cls_atten_d2], dim=2)
                    gt_atten0 = torch.cat([gt_atten0, cls_atten_d1], dim=1)
                gt_atten0 = gt_atten0.repeat([head_i, 1, 1, 1]).transpose(0, 1)
                gt_atten.append(gt_atten0)

            else:
                mask_w0 = rearrange(mask[:, :, :h_block, :w_block], 'b c (b1 h) (b2 w) -> (b b1 b2) (c h w) 1',
                                    h=window_i, w=window_i)
                mask_w1 = rearrange(mask[:, :, h_block - h:, w_block - w:], 'b c (b1 h) (b2 w) -> (b b1 b2) (c h w) 1',
                                    h=window_i, w=window_i)
                gt_w0 = rearrange(gt_tmp[:, :, :h_block, :w_block], 'b c (b1 h) (b2 w) -> (b b1 b2) (c h w) 1',
                                  h=window_i, w=window_i)
                gt_w1 = rearrange(gt_tmp[:, :, h_block - h:, w_block - w:], 'b c (b1 h) (b2 w) -> (b b1 b2) (c h w) 1',
                                  h=window_i, w=window_i)
                c_merge_w0 = rearrange(c_merge[:, :, :h_block, :w_block], 'b c (b1 h) (b2 w) -> (b b1 b2) (c h w) 1',
                                       h=window_i, w=window_i)
                c_merge_w1 = rearrange(c_merge[:, :, h_block - h:, w_block - w:],
                                       'b c (b1 h) (b2 w) -> (b b1 b2) (c h w) 1', h=window_i, w=window_i)
                gt_c0 = (gt_w0 @ c_merge_w0.transpose(-2, -1)) / self.tempt
                gt_atten0 = torch.nn.functional.softmax(gt_c0, dim=-1) * mask_w0
                gt_c1 = (gt_w1 @ c_merge_w1.transpose(-2, -1)) / self.tempt
                gt_atten1 = torch.nn.functional.softmax(gt_c1, dim=-1) * mask_w1
                if cls_i and not self.clsignore and not self.clsattenignore:
                    hw, win2, temwin2 = gt_atten0.shape
                    avr_va = 1 / (temwin2 + 1)
                    mask_w0_cls = torch.sum(gt_atten0, dim=-1, keepdim=True)
                    mask_w1_cls = torch.sum(gt_atten1, dim=-1, keepdim=True)
                    gt_atten0 = gt_atten0 * temwin2 / (temwin2 + 1)
                    gt_atten1 = gt_atten1 * temwin2 / (temwin2 + 1)
                    cls_atten_d2a0 = avr_va * mask_w0_cls * torch.ones([hw, win2, 1]).cuda()
                    cls_atten_d2a1 = avr_va * mask_w1_cls * torch.ones([hw, win2, 1]).cuda()
                    cls_atten_d1 = avr_va * self.neg2posweight * num_positive / (num_positive + num_negative) * torch.ones([hw, 1, temwin2 + 1]).cuda()
                    gt_atten0 = torch.cat([gt_atten0, cls_atten_d2a0], dim=2)
                    gt_atten0 = torch.cat([gt_atten0, cls_atten_d1], dim=1)
                    gt_atten1 = torch.cat([gt_atten1, cls_atten_d2a1], dim=2)
                    gt_atten1 = torch.cat([gt_atten1, cls_atten_d1], dim=1)
                gt_atten0 = gt_atten0.repeat([head_i, 1, 1, 1]).transpose(0, 1)
                gt_atten1 = gt_atten1.repeat([head_i, 1, 1, 1]).transpose(0, 1)
                gt_atten.append(gt_atten0)
                gt_atten.append(gt_atten1)

            gt_attens.append(gt_atten)
            gt = self.build_maxpooling(gt)  # for
            #gt_fuz = self.build_maxpooling(gt_fuz)

        return gt_attens

# datasets
class Signle_Loader(data.Dataset):
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False, split_value = 127.5):
        self.root = root
        self.split = split
        self.transform = transform
        self.split_value = split_value
        if self.split == 'train':
            self.filelist = join(self.root, 'bsds_pascal_train_pair.lst')
        elif self.split == 'test':
            self.filelist = join(self.root + '/HED-BSDS', 'test.lst')
            self.root = self.root + '/HED-BSDS'
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < self.split_value)] = 2
            lb[lb >= self.split_value] = 1


        else:
            img_file = self.filelist[index].rstrip()

        if self.split == "train":
            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_cv2(img)
            return img, lb
        else:
            img = np.array(Image.open(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_PIL(img)
            return img, img_file.split('/')[-1].split('.')[0]

class Multi_Loader(data.Dataset):

    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False, split_value = 127.5, datasetname = ''):
        self.root = root
        self.split = split
        self.transform = transform
        self.split_value = split_value
        if self.split == 'train':
            self.filelist = join(self.root, datasetname + '.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, datasetname + '-test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file, dataset_id = self.filelist[index].split()
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < self.split_value)] = 2
            lb[lb >= self.split_value] = 1


        else:
            img_file, dataset_id = self.filelist[index].split()

        if self.split == "train":
            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_cv2(img)
            return img, lb, dataset_id, img_file, lb_file
        else:
            img = np.array(Image.open(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_PIL(img)
            return img, dataset_id, img_file.split('/')[-1].split('.')[0]