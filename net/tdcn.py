import numpy as np
import torch
from torch import nn
from einops import rearrange
import math
import torch.nn.functional as F

# layers or sub-blocks
def Aggregate():
    return nn.Sequential(
        nn.MaxPool2d(2, stride=2, ceil_mode=True)
    )

def Aggregate_dim(dim_in, dim_out):
    if dim_in != dim_out:
        return nn.Sequential(
            LayerNorm_conv(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
    else:
        return nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

def interop2(data1, th, tw):
    _, _, h1, w1 = data1.size()
    data1 = torch.nn.functional.interpolate(data1, size=(th, tw), mode='bilinear', align_corners=False)
    return data1

class LayerNorm_conv(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

def config_op():
    def createConvFunc(op_type):
        assert op_type in ['cd', 'ad', 'hd', 'sd'], 'unknown op type: %s' % str(op_type)
        # Laplace
        if op_type == 'cd':
            def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
                assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
                assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
                assert padding == dilation, 'padding for cd_conv set wrong'

                weights_c = weights.sum(dim=[2, 3], keepdim=True)
                yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
                y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
                return y - yc

            return func
        # Texture
        elif op_type == 'ad':
            def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
                assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
                assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
                assert padding == dilation, 'padding for ad_conv set wrong'

                shape = weights.shape
                weights = weights.view(shape[0], shape[1], -1)
                weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
                y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
                return y

            return func
        # Sobel x&y
        elif op_type == 'hd':
            def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
                assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
                assert weights.size(2) == 1 and weights.size(3) == 3, 'kernel size for hd_conv should be 3x3'
                assert padding == dilation, 'padding for cd_conv set wrong'

                shape = weights.shape
                if weights.is_cuda:
                    buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 3).fill_(0)
                else:
                    buffer = torch.zeros(shape[0], shape[1], 3 * 3)
                weights = weights.view(shape[0], shape[1], -1)
                buffer[:, :, [0, 2]] = weights[:, :, [0, 2]]
                buffer[:, :, [6, 8]] = - weights[:, :, [0, 2]]
                buffer[:, :, [1]] = 2 * weights[:, :, [1]]
                buffer[:, :, [7]] = - 2 * weights[:, :, [1]]
                buffer = buffer.view(shape[0], shape[1], 3, 3)
                y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
                return y

            return func
        elif op_type == 'sd':
            def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
                assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
                assert weights.size(2) == 3 and weights.size(3) == 1, 'kernel size for hd_conv should be 3x3'
                assert padding == dilation, 'padding for cd_conv set wrong'

                shape = weights.shape
                if weights.is_cuda:
                    buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 3).fill_(0)
                else:
                    buffer = torch.zeros(shape[0], shape[1], 3 * 3)
                weights = weights.view(shape[0], shape[1], -1)
                buffer[:, :, [0, 6]] = weights[:, :, [0, 2]]
                buffer[:, :, [2, 8]] = - weights[:, :, [0, 2]]
                buffer[:, :, [3]] = 2 * weights[:, :, [1]]
                buffer[:, :, [5]] = - 2 * weights[:, :, [1]]
                buffer = buffer.view(shape[0], shape[1], 3, 3)
                y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
                return y

            return func
        else:
            return None

    edge_op = ['cd', 'ad', 'hd', 'sd']
    pdcs = []
    for i in range(4):
        pdcs.append(createConvFunc(edge_op[i]))
    return pdcs

# Thanks https://github.com/zhuoinoulu/pidinet.
class TAG(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, kenel_method=0):
        super(TAG, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if kenel_method == 0:
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        elif kenel_method == 1:
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, 1, kernel_size))
        elif kenel_method == 2:
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return torch.abs(self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups))

class BAtten(nn.Module):
    def __init__(self, dim, template=13, heads=8, dropout=0., cls=False, window=5, pos_embbing=False, bias=False,
                 pos_embbding_mode='absolute', noone=False, clsignore=True, clsattenignore=True,
                 neg_val=-0.5, pos_val=1 + 1 / 3, nor_val=-3 / 8, cnor_val=4,
                 tempt=1):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.window = window
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.tempt = tempt
        self.clsignore = clsignore
        self.clsattenignore = clsattenignore
        self.to_k = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.filt = []

        if template == 1 or not noone:
            # if template > 0:
            self.tw_0 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=None)
            nw_0 = np.array([nor_val, nor_val, nor_val, nor_val, cnor_val, nor_val, nor_val, nor_val, nor_val]).reshape(
                [3, 3])
            self.tw_0.weight = self.build_weight(nw_0, dim)
            self.filt.append(self.tw_0)

        if template > 1:
            self.tw_1 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=None)
            nw_1 = np.array([neg_val, pos_val, neg_val, neg_val, pos_val, neg_val, neg_val, pos_val, neg_val]).reshape(
                [3, 3])
            self.tw_1.weight = self.build_weight(nw_1, dim)
            self.tw_2 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=None)
            nw_2 = np.array([neg_val, neg_val, neg_val, pos_val, pos_val, pos_val, neg_val, neg_val, neg_val]).reshape(
                [3, 3])
            self.tw_2.weight = self.build_weight(nw_2, dim)
            self.filt.append(self.tw_1)
            self.filt.append(self.tw_2)

        if template > 2:
            self.tw_3 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=None)
            nw_3 = np.array([pos_val, neg_val, neg_val, neg_val, pos_val, neg_val, neg_val, neg_val, pos_val]).reshape(
                [3, 3])
            self.tw_3.weight = self.build_weight(nw_3, dim)
            self.tw_4 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=None)
            nw_4 = np.array([neg_val, neg_val, pos_val, neg_val, pos_val, neg_val, pos_val, neg_val, neg_val]).reshape(
                [3, 3])
            self.tw_4.weight = self.build_weight(nw_4, dim)
            self.filt.append(self.tw_3)
            self.filt.append(self.tw_4)

        if template > 4:
            self.tw_5 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=None)
            nw_5 = np.array([neg_val, pos_val, neg_val, pos_val, pos_val, neg_val, neg_val, neg_val, neg_val]).reshape(
                [3, 3])
            self.tw_5.weight = self.build_weight(nw_5, dim)
            self.tw_6 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=None)
            nw_6 = np.array([neg_val, pos_val, neg_val, neg_val, pos_val, pos_val, neg_val, neg_val, neg_val]).reshape(
                [3, 3])
            self.tw_6.weight = self.build_weight(nw_6, dim)
            self.tw_7 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=None)
            nw_7 = np.array([neg_val, neg_val, neg_val, pos_val, pos_val, neg_val, neg_val, pos_val, neg_val]).reshape(
                [3, 3])
            self.tw_7.weight = self.build_weight(nw_7, dim)
            self.tw_8 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=None)
            nw_8 = np.array([neg_val, neg_val, neg_val, neg_val, pos_val, pos_val, neg_val, pos_val, neg_val]).reshape(
                [3, 3])
            self.tw_8.weight = self.build_weight(nw_8, dim)
            self.filt.append(self.tw_5)
            self.filt.append(self.tw_6)
            self.filt.append(self.tw_7)
            self.filt.append(self.tw_8)

        self.cls = cls
        if self.cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.filt_num = len(self.filt)
        self.pos_embbing = pos_embbing
        self.pos_embbding_mode = pos_embbding_mode
        if pos_embbing and pos_embbding_mode == 'absolute':
            self.pos_emb_kv = nn.Parameter(torch.zeros(1, window * window * self.filt_num, dim))
            self.pos_emb_q = nn.Parameter(torch.zeros(1, window * window, dim))
        if pos_embbing and pos_embbding_mode == 'relative':
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window - 1) * (2 * window - 1), heads))  # 2*Wh-1 * 2*Ww-1, nH
            coords_h = torch.arange(self.window)
            coords_w = torch.arange(self.window)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window - 1
            relative_coords[:, :, 0] *= 2 * self.window - 1
            self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

    def forward(self, x):
        b, c, h, w = x.shape
        heads = self.heads
        h_block = int(np.ceil(h / self.window) * self.window)
        w_block = int(np.ceil(w / self.window) * self.window)
        attns = []
        if h_block == h and w_block == w:
            pose_merge = []
            for f_i, filt in enumerate(self.filt):
                kv = filt(x)
                if f_i == 0:
                    pose_merge = kv
                else:
                    pose_merge = torch.cat([pose_merge, kv], dim=1)
            kv_0 = rearrange(pose_merge, 'b (num c) (b1 h) (b2 w) -> (b b1 b2) (num h w) c', h=self.window,
                             w=self.window, num=self.filt_num)
            q_0 = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) (h w) c', h=self.window, w=self.window)
            if self.pos_embbing and self.pos_embbding_mode == 'absolute':
                kv_0 = kv_0 + self.pos_emb_kv
                q_0 = q_0 + self.pos_emb_q
            if self.cls:
                cls = self.cls_token.repeat([int(h_block * w_block / self.window ** 2), 1, 1])
                kv_0 = torch.cat([kv_0, cls], dim=1)
                q_0 = torch.cat([q_0, cls], dim=1)

            b_, n_, c_ = q_0.shape
            k_bias_0 = self.to_k(kv_0)
            v_bias_0 = self.to_v(kv_0)
            q0 = self.to_q(q_0)
            k_bias_0 = rearrange(k_bias_0, 'b x (h d) -> b h x d', h=heads)
            v_bias_0 = rearrange(v_bias_0, 'b x (h d) -> b h x d', h=heads)
            q0 = rearrange(q0, 'b x (h d) -> b h x d', h=heads)
            q0 = q0 * self.scale

            # q0_norm = torch.norm(q0, dim=-1, keepdim=True)
            # q0 = q0 / q0_norm
            # k_bias_0_norm = torch.norm(k_bias_0, dim=-1, keepdim=True)
            # k_bias_0 = k_bias_0 / k_bias_0_norm

            attn0 = (q0 @ k_bias_0.transpose(-2, -1))
            if self.pos_embbing and self.pos_embbding_mode == 'relative':
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window * self.window, self.window * self.window, -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                relative_position_bias = relative_position_bias.repeat([1, 1, self.filt_num])
                if self.cls:
                    attn0[:, :, :-1, :-1] = attn0[:, :, :-1, :-1] + relative_position_bias.unsqueeze(0)
                else:
                    attn0 = attn0 + relative_position_bias.unsqueeze(0)
            if self.clsignore and self.cls:
                attn0_out0 = torch.ones(attn0[:, :, :-1, :-1].shape).cuda()
            elif self.clsattenignore and self.cls:
                attn0_out0 = self.attend(attn0[:, :, :-1, :-1] / self.tempt)
            else:
                attn0_out0 = self.attend(attn0 / self.tempt)
            attn0 = self.attend(attn0)
            attns.append(attn0_out0)

            '''
            attn0_4 = rearrange(attn0, '(h w) head b t -> (b head) t h w', h=int(h / self.window), w =int(w / self.window))
            attn0_4_ = attn0_4.detach().cpu().numpy()[0]
            attn0_4_t0 = attn0_4_[0]
            attn0_4_t1 = attn0_4_[1]
            attn0_4_t2 = attn0_4_[2]
            attn0_4_t3 = attn0_4_[3]
            #attn0_4_t4 = attn0_4_[4]
            #attn0_4_t4 = attn0_4_[4]
            #attn0_4_t5 = attn0_4_[5]
            #attn0_4_t6 = attn0_4_[6]
            #attn0_4_t7 = attn0_4_[7]
            #attn0_4_t8 = attn0_4_[8]
            #po1 = attn0_4_[:,112,133]
            #po2 = attn0_4_[:,107,137]
            po11 = attn0_4_[:,28,33]
            po21 = attn0_4_[:,27,34]
            sp0 = np.mean(attn0_4_t0)
            sp1 = np.mean(attn0_4_t1)
            sp2 = np.mean(attn0_4_t2)
            sp3 = np.mean(attn0_4_t3)
            #sp4 = np.mean(attn0_4_t4)
            '''

            out0 = (attn0 @ v_bias_0).transpose(1, 2)

            out0 = self.to_out(out0.reshape(b_, n_, c_))
            if self.cls:
                x_0 = rearrange(out0[:, :-1, :], '(b b1 b2) (h w) c -> b c (b1 h) (b2 w)',
                                b1=int(h_block / self.window),
                                b2=int(w_block / self.window), h=self.window, w=self.window)
            else:
                x_0 = rearrange(out0, '(b b1 b2) (h w) c -> b c (b1 h) (b2 w)',
                                b1=int(h_block / self.window),
                                b2=int(w_block / self.window), h=self.window, w=self.window)
            out = x_0

        else:
            x = nn.ReflectionPad2d((w_block - w, w_block - w, h_block - h, h_block - h))(x)
            pose_merge = []
            for f_i, filt in enumerate(self.filt):
                kv = filt(x)
                if f_i == 0:
                    pose_merge = kv
                else:
                    pose_merge = torch.cat([pose_merge, kv], dim=1)

            kv_0 = pose_merge[:, :, :h_block, :w_block]
            kv_1 = pose_merge[:, :, h_block - h:, w_block - w:]
            kv_0 = rearrange(kv_0, 'b (num c) (b1 h) (b2 w) -> (b b1 b2) (num h w) c', h=self.window, w=self.window,
                             num=self.filt_num)
            kv_1 = rearrange(kv_1, 'b (num c) (b1 h) (b2 w) -> (b b1 b2) (num h w) c', h=self.window, w=self.window,
                             num=self.filt_num)
            q_0 = rearrange(x[:, :, :h_block, :w_block], 'b c (b1 h) (b2 w) -> (b b1 b2) (h w) c', h=self.window,
                            w=self.window)
            q_1 = rearrange(x[:, :, h_block - h:, w_block - w:], 'b c (b1 h) (b2 w) -> (b b1 b2) (h w) c',
                            h=self.window, w=self.window)

            if self.pos_embbing and self.pos_embbding_mode == 'absolute':
                kv_0 = kv_0 + self.pos_emb_kv
                kv_1 = kv_1 + self.pos_emb_kv
                q_0 = q_0 + self.pos_emb_q
                q_1 = q_1 + self.pos_emb_q

            if self.cls:
                cls = self.cls_token.repeat([int(h_block * w_block / self.window ** 2), 1, 1])
                kv_0 = torch.cat([kv_0, cls], dim=1)
                kv_1 = torch.cat([kv_1, cls], dim=1)
                q_0 = torch.cat([q_0, cls], dim=1)
                q_1 = torch.cat([q_1, cls], dim=1)

            b_, n_, c_ = q_0.shape

            k_bias_0 = self.to_k(kv_0)
            k_bias_1 = self.to_k(kv_1)
            v_bias_0 = self.to_v(kv_0)
            v_bias_1 = self.to_v(kv_1)
            q0 = self.to_q(q_0)
            q1 = self.to_q(q_1)

            k_bias_0 = rearrange(k_bias_0, 'b x (h d) -> b h x d', h=heads)
            k_bias_1 = rearrange(k_bias_1, 'b x (h d) -> b h x d', h=heads)
            v_bias_0 = rearrange(v_bias_0, 'b x (h d) -> b h x d', h=heads)
            v_bias_1 = rearrange(v_bias_1, 'b x (h d) -> b h x d', h=heads)
            q0 = rearrange(q0, 'b x (h d) -> b h x d', h=heads)
            q0 = q0 * self.scale
            q1 = rearrange(q1, 'b x (h d) -> b h x d', h=heads)
            q1 = q1 * self.scale

            # q0_norm = torch.norm(q0, dim=-1, keepdim=True)
            # q0 = q0 / q0_norm
            # k_bias_0_norm = torch.norm(k_bias_0, dim=-1, keepdim=True)
            # k_bias_0 = k_bias_0 / k_bias_0_norm

            # q1_norm = torch.norm(q1, dim=-1, keepdim=True)
            # q1 = q1 / q1_norm
            # k_bias_1_norm = torch.norm(k_bias_1, dim=-1, keepdim=True)
            # k_bias_1 = k_bias_1 / k_bias_1_norm

            attn0 = (q0 @ k_bias_0.transpose(-2, -1))
            attn1 = (q1 @ k_bias_1.transpose(-2, -1))
            if self.pos_embbing and self.pos_embbding_mode == 'relative':
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window * self.window, self.window * self.window, -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                relative_position_bias = relative_position_bias.repeat([1, 1, self.filt_num])
                if self.cls:
                    attn0[:, :, :-1, :-1] = attn0[:, :, :-1, :-1] + relative_position_bias.unsqueeze(0)
                    attn1[:, :, :-1, :-1] = attn1[:, :, :-1, :-1] + relative_position_bias.unsqueeze(0)
                else:
                    attn0 = attn0 + relative_position_bias.unsqueeze(0)
                    attn1 = attn1 + relative_position_bias.unsqueeze(0)

            if self.clsignore and self.cls:
                attn0_out0 = torch.ones(attn0[:, :, :-1, :-1].shape).cuda()
                attn1_out1 = torch.ones(attn1[:, :, :-1, :-1].shape).cuda()
            elif self.clsattenignore and self.cls:
                attn0_out0 = self.attend(attn0[:, :, :-1, :-1] / self.tempt)
                attn1_out1 = self.attend(attn1[:, :, :-1, :-1] / self.tempt)
            else:
                attn0_out0 = self.attend(attn0 / self.tempt)
                attn1_out1 = self.attend(attn1 / self.tempt)

            attn0 = self.attend(attn0)
            out0 = (attn0 @ v_bias_0).transpose(1, 2)

            attn1 = self.attend(attn1)
            out1 = (attn1 @ v_bias_1).transpose(1, 2)

            attns.append(attn0_out0)
            attns.append(attn1_out1)

            out0 = self.to_out(out0.reshape(b_, n_, c_))
            out1 = self.to_out(out1.reshape(b_, n_, c_))

            if self.cls:
                x_0 = rearrange(out0[:, :-1, :], '(b b1 b2) (h w) c -> b c (b1 h) (b2 w)',
                                b1=int(h_block / self.window),
                                b2=int(w_block / self.window), h=self.window, w=self.window)
                x_1 = rearrange(out1[:, :-1, :], '(b b1 b2) (h w) c -> b c (b1 h) (b2 w)',
                                b1=int(h_block / self.window),
                                b2=int(w_block / self.window), h=self.window, w=self.window)
            else:
                x_0 = rearrange(out0, '(b b1 b2) (h w) c -> b c (b1 h) (b2 w)',
                                b1=int(h_block / self.window),
                                b2=int(w_block / self.window), h=self.window, w=self.window)
                x_1 = rearrange(out1, '(b b1 b2) (h w) c -> b c (b1 h) (b2 w)',
                                b1=int(h_block / self.window),
                                b2=int(w_block / self.window), h=self.window, w=self.window)
            out = (x_0[:, :, h_block - h:, w_block - w:] + x_1[:, :, 0:h, 0:w]) / 2

        return out, attns

    def build_weight(self, wei_np, dim):
        ws_i = torch.from_numpy(wei_np).unsqueeze(0).unsqueeze(0).float()
        kernel = np.repeat(ws_i, dim, axis=0)
        return torch.nn.Parameter(kernel, requires_grad=False)

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult=4, dropout=0., trans_activate=nn.GELU(), bias=True):
        super().__init__()
        self.da = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_mult), 1, bias=bias),
        )

        self.db = nn.Sequential(
            LayerNorm_conv(int(dim * mlp_mult)),
            nn.Conv2d(int(dim * mlp_mult), int(dim * mlp_mult), 3, bias=bias, padding=1, groups=int(dim * mlp_mult)),
        )

        self.dc = nn.Sequential(
            trans_activate,
            nn.Conv2d(int(dim * mlp_mult), dim, 1, bias=bias),
        )

    def forward(self, x):
        x = self.da(x)
        x = self.dc(self.db(x) + x)
        return x

class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.bn1 = LayerNorm_conv(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.bn2 = LayerNorm_conv(out_channels)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=3, padding=3)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7)
        self.bn2_1 = LayerNorm_conv(out_channels)
        self.bn2_2 = LayerNorm_conv(out_channels)
        self.bn2_3 = LayerNorm_conv(out_channels)
        self.bn2_4 = LayerNorm_conv(out_channels)
        self.relu2_1 = nn.GELU()
        self.relu2_2 = nn.GELU()
        self.relu2_3 = nn.GELU()
        self.relu2_4 = nn.GELU()

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn3 = LayerNorm_conv(out_channels)
        self.relu3 = nn.GELU()

    def forward(self, x):
        x = self.conv1(self.bn1(x))

        y = self.bn2(x)
        x1 = self.relu2_1(self.bn2_1(self.conv2_1(y)))
        x2 = self.relu2_2(self.bn2_2(self.conv2_2(y)))
        x3 = self.relu2_3(self.bn2_3(self.conv2_3(y)))
        x4 = self.relu2_4(self.bn2_4(self.conv2_4(y)))
        return self.bn3(self.conv3(x1 + x2 + x3 + x4) + x)

# blocks
class DConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, depth, cnn_activate, pdcs, bn=True, bias=False, ):
        super().__init__()
        self.initlayer = nn.ModuleList([])
        self.dblock1_layer = nn.ModuleList([])
        self.merge1_block_layer = nn.ModuleList([])
        self.dblock2_layer = nn.ModuleList([])
        self.bn = bn
        self.depth = depth
        self.dim_in = dim_in
        self.filt = []

        for i in range(depth):
            initlayer = nn.ModuleList([])
            dblock1_layers = nn.ModuleList([])
            merge1_block_layers = nn.ModuleList([])
            dblock2_layers = nn.ModuleList([])
            #
            if dim_in != dim_out and dim_in != 3:
                ini_short = nn.Sequential(LayerNorm_conv(dim_in),
                                          nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0), )
            else:
                ini_short = nn.Identity()
            if dim_in == 3:
                initlayer.append(nn.ModuleList([ini_short,

                                                nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
                                                ]))
            else:
                initlayer.append(nn.ModuleList([
                    ini_short,
                    LayerNorm_conv(dim_in),

                    nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1, groups=dim_in),
                    LayerNorm_conv(dim_out),
                    cnn_activate,

                    nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=2, groups=dim_in, dilation=2),
                    LayerNorm_conv(dim_out),
                    cnn_activate,

                    nn.Conv2d(dim_out, dim_out, kernel_size=1, padding=0, bias=bias), ]))

            dim_in = dim_out
            shortcut = nn.Identity()
            dblock1_layers.append(nn.ModuleList([shortcut,
                                                 LayerNorm_conv(dim_in),

                                                 TAG(pdcs[0], dim_in, dim_in, kernel_size=3, padding=1,
                                                        groups=dim_in,
                                                        bias=bias),
                                                 LayerNorm_conv(dim_in),
                                                 cnn_activate,

                                                 TAG(pdcs[1], dim_in, dim_in, kernel_size=3, padding=1,
                                                        groups=dim_in,
                                                        bias=bias),
                                                 LayerNorm_conv(dim_in),
                                                 cnn_activate,

                                                 TAG(pdcs[0], dim_in, dim_in, kernel_size=3, padding=2,
                                                        groups=dim_in, dilation=2,
                                                        bias=bias),
                                                 LayerNorm_conv(dim_in),
                                                 cnn_activate,

                                                 TAG(pdcs[1], dim_in, dim_in, kernel_size=3, padding=2,
                                                        groups=dim_in, dilation=2,
                                                        bias=bias),
                                                 LayerNorm_conv(dim_in),
                                                 cnn_activate,

                                                 nn.Conv2d(dim_out, dim_out, kernel_size=1, padding=0, bias=bias),
                                                 ]))

            merge1_block_layers.append(nn.ModuleList([shortcut,
                                                      LayerNorm_conv(dim_in),

                                                      nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1,
                                                                groups=dim_in, bias=bias),
                                                      LayerNorm_conv(dim_out),
                                                      cnn_activate,

                                                      nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=2,
                                                                groups=dim_in, dilation=2),
                                                      LayerNorm_conv(dim_out),
                                                      cnn_activate,

                                                      nn.Conv2d(dim_out, dim_out, kernel_size=1, padding=0, bias=bias),
                                                      ]))

            dblock2_layers.append(nn.ModuleList([shortcut,
                                                 LayerNorm_conv(dim_in),

                                                 TAG(pdcs[2], dim_in, dim_in, kernel_size=3, padding=1,
                                                        groups=dim_in, bias=bias, kenel_method=1),
                                                 LayerNorm_conv(dim_in),
                                                 cnn_activate,

                                                 TAG(pdcs[3], dim_in, dim_in, kernel_size=3, padding=1,
                                                        groups=dim_in, bias=bias, kenel_method=2),
                                                 LayerNorm_conv(dim_in),
                                                 cnn_activate,

                                                 TAG(pdcs[2], dim_in, dim_in, kernel_size=3, padding=2,
                                                        groups=dim_in, dilation=2, bias=bias, kenel_method=1),
                                                 LayerNorm_conv(dim_in),
                                                 cnn_activate,

                                                 TAG(pdcs[3], dim_in, dim_in, kernel_size=3, padding=2,
                                                        groups=dim_in, dilation=2, bias=bias, kenel_method=2),
                                                 LayerNorm_conv(dim_in),
                                                 cnn_activate,

                                                 nn.Conv2d(dim_out, dim_out, kernel_size=1, padding=0, bias=bias),
                                                 ]))
            self.initlayer.append(initlayer)
            self.dblock1_layer.append(dblock1_layers)
            self.merge1_block_layer.append(merge1_block_layers)
            self.dblock2_layer.append(dblock2_layers)

    def forward(self, x):
        fea = []
        _, c, h, w = x.shape
        for idx in range(self.depth):
            initlayer_ = self.initlayer[idx]
            dblock1_layer_ = self.dblock1_layer[idx]
            merge1_block_layer_ = self.merge1_block_layer[idx]
            dblock2_layer_ = self.dblock2_layer[idx]

            if self.dim_in == 3:
                for s1, c1 in initlayer_:
                    x = c1(x)
            else:
                for s1, b0, c11, b11, r11, c21, b21, r21, cm in initlayer_:
                    y = b0(x)
                    x = cm(r11(b11(c11(y))) + r21(b21(c21(y)))) + s1(x)

            for s1, b0, c11, b11, r11, c21, b21, r21, c31, b31, r31, c41, b41, r41, cm in dblock1_layer_:
                y = b0(x)
                x = cm(r11(b11(c11(y))) + r21(b21(c21(y))) +
                       r31(b31(c31(y))) + r41(b41(c41(y)))) + s1(x)

            for s1, b0, c11, b11, r11, c21, b21, r21, cm in merge1_block_layer_:
                y = b0(x)
                x = cm(r11(b11(c11(y))) + r21(b21(c21(y)))) + s1(x)

            for s1, b0, c11, b11, r11, c21, b21, r21, c31, b31, r31, c41, b41, r41, cm in dblock2_layer_:
                y = b0(x)
                x = cm(r11(b11(c11(y))) + r21(b21(c21(y))) +
                       r31(b31(c31(y))) + r41(b41(c41(y)))) + s1(x)

            fea.append(x)
        return x, fea

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, template, mlp_mult, window_size, cls=False, dropout=0., bn=False,
                 pos_embbing=False, bias=False, pos_embbding_mode='absolute', trans_activate=nn.GELU(),
                 noone=False, clsignore=True, clsattenignore=True, neg_val=-0.5, pos_val=1 + 1 / 3, nor_val=-3 / 8,
                 cnor_val=4, tempt=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.window_size = window_size
        self.BN = bn
        self.out_atten = False
        self.general = False
        self.out_atten = True
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LayerNorm_conv(dim),
                BAtten(dim, template=template, heads=heads, dropout=dropout, cls=cls,
                      window=window_size, pos_embbing=pos_embbing, bias=bias,
                      pos_embbding_mode=pos_embbding_mode,
                      noone=noone, clsignore=clsignore, clsattenignore=clsattenignore,
                      neg_val=neg_val, pos_val=pos_val, nor_val=nor_val, cnor_val=cnor_val, tempt=tempt),
                LayerNorm_conv(dim),
                FeedForward(dim, mlp_mult, dropout=dropout, trans_activate=trans_activate, bias=bias),
            ]))

    def forward(self, x):
        fea = []
        for ly1, attn, ly2, ff in self.layers:
            y, attenout = attn(ly1(x))
            x = y + x
            y = ff(ly2(x))
            x = y + x
            fea.append(x)
        return x, fea, attenout

class Head(nn.Module):
    def __init__(self, input_dim_list, fea_dim, stride, head_mode='cat', ):
        super().__init__()

        self.conv1_down = ASPP(input_dim_list[0], fea_dim)
        self.conv2_down = ASPP(input_dim_list[1], fea_dim)
        self.conv3_down = ASPP(input_dim_list[2], fea_dim)
        self.conv4_down = ASPP(input_dim_list[3], fea_dim)

        self.conv1_downf = nn.Identity()
        self.conv2_downf = nn.Identity()
        self.conv3_downf = nn.Identity()
        self.conv4_downf = nn.Identity()

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(fea_dim, 1, 1)
        self.score_dsn2 = nn.Conv2d(fea_dim, 1, 1)
        self.score_dsn3 = nn.Conv2d(fea_dim, 1, 1)
        self.score_dsn4 = nn.Conv2d(fea_dim, 1, 1)

        self.head_mode = head_mode
        # lr 0.001 0.002 decay 1 0
        # self.score_final = nn.Conv2d(4, 1, 1)
        self.stride = stride
        self.soft = torch.nn.functional.softmax

    def forward(self, fea, img_H, img_W):
        conv1_1_down = self.conv1_downf(self.conv1_down(fea[0][0]))
        conv2_1_down = self.conv2_downf(self.conv2_down(fea[1][0]))
        conv3_1_down = self.conv3_downf(self.conv3_down(fea[2][0]))
        conv4_1_down = self.conv4_downf(self.conv4_down(fea[3][0]))

        so11_out = self.score_dsn1(conv1_1_down)
        so21_out = self.score_dsn2(conv2_1_down)
        so31_out = self.score_dsn3(conv3_1_down)
        so41_out = self.score_dsn4(conv4_1_down)

        so11 = so11_out
        so21 = interop2(so21_out, img_H, img_W)
        so31 = interop2(so31_out, img_H, img_W)
        so41 = interop2(so41_out, img_H, img_W)

        # dot mode

        fuse = (so11 + so21 + so31 + so41) / 4
        # end mode

        results = [so11, so21, so31, so41, fuse.detach()]
        results = [torch.sigmoid(r) for r in results]
        return results


# main network
class TDCN(nn.Module):
    def __init__(
            self,
            *,
            templates,
            dim,
            heads,
            window_size,
            cnn_repeats,
            block_repeats,
            stride,
            pos_embbing,
            cls,
            mlp_mult=4,
            dropout=0.,
            bn=False,
            head_mode='cat',
            bias=True,
            pos_embbding_mode='relative',
            trans_activate=nn.GELU(),
            cnn_activate=nn.ReLU(),
            fea_dim_head=24,
            cnn_bn=False,
            cnn_bias=False,
            noone=True,
            clsignore=False,
            clsattenignore=True,
            neg_val=-0.5,
            pos_val=1 + 1 / 3,
            nor_val=-3 / 8,
            cnor_val=4,
            tempt=1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.window_size = window_size
        self.templates = templates
        self.heads = heads
        self.cls = cls

        pdcs = config_op()

        for level in range(len(dim)):
            if level == 0:
                self.layers.append(nn.ModuleList([
                    DConvBlock(3, dim[level], cnn_repeats[level], cnn_activate, pdcs, bn=cnn_bn,
                              bias=cnn_bias, ),
                    Transformer(dim[level], block_repeats[level], heads[level], templates[level], mlp_mult,
                                window_size[level], cls[level], dropout, bn, pos_embbing[level], bias,
                                pos_embbding_mode, trans_activate,
                                noone=noone, clsignore=clsignore, clsattenignore=clsattenignore,
                                neg_val=neg_val, pos_val=pos_val, nor_val=nor_val, cnor_val=cnor_val, tempt=tempt),
                    Aggregate(),
                    nn.Identity()
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    DConvBlock(dim[level - 1], dim[level], cnn_repeats[level], cnn_activate, pdcs, bn=cnn_bn,
                              bias=cnn_bias, ),
                    Transformer(dim[level], block_repeats[level], heads[level], templates[level], mlp_mult,
                                window_size[level], cls[level], dropout, bn, pos_embbing[level], bias,
                                pos_embbding_mode, trans_activate,
                                noone=noone, clsignore=clsignore, clsattenignore=clsattenignore,
                                neg_val=neg_val, pos_val=pos_val, nor_val=nor_val, cnor_val=cnor_val, tempt=tempt),
                    Aggregate() if not level == len(dim) - 1 else nn.Identity(),
                    Aggregate_dim(dim[level - 1], dim[level])
                ]))

        self.bound_head = Head(input_dim_list=dim, fea_dim=fea_dim_head,
                               stride=stride, head_mode=head_mode, )

    def forward(self, img):
        _, _, h_raw, w_raw = img.shape
        featurelist = []
        attenlist = []
        x = img
        num_hierarchies = len(self.layers)

        for level, (conv, transformer, aggregate, aggregate_t) in zip(range(num_hierarchies), self.layers):
            x, fea_c = conv(x)
            # if level == 0:
            #    fea_cc = fea_c

            if level == 0:
                _, fea_t, attenout = transformer(x)
            else:
                _, fea_t, attenout = transformer(x + aggregate_t(featurelist[level - 1][0]))
            featurelist.append(fea_t + fea_c)
            attenlist.append(attenout)
            x = aggregate(x)
        bound_out = self.bound_head(featurelist, h_raw, w_raw)
        return bound_out, featurelist, attenlist