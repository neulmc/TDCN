import numpy as np
import torch
from net.tdcn_deploy import TDCN
import torch.optim as optim
from net.data_loader import gt2attengt, Signle_Loader
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import argparse

def build_model():
    nest = TDCN(
        templates=[4, 4, 2, 1],
        dim=[42, 84, 168, 168],
        #dim=[6, 12, 24, 24],
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

def build_input(filename):
    img = np.array(Image.open(filename), dtype=np.float32)
    img = img[:, :, ::-1] - np.zeros_like(img)  # rgb to bgr
    img -= np.array((104.00698793, 116.66876762, 122.67891434))
    preprocess = transforms.Compose([transforms.ToTensor(),])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# for BSDS
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-filename", default='demo.jpg', type=str, help="File name")
    parser.add_argument("-head_idx", default=4, type=int, help="Selected head function (between 1-4)")
    parser.add_argument("--fusion_mode", action='store_true', help="Whether to use fusion prediction of different scale head functions")
    parser.add_argument("--cuda_available", action='store_true', help="Whether to use GPU")

    args = parser.parse_args()
    filename = args.filename
    head_idx = args.head_idx
    fusion_mode = args.fusion_mode
    cuda_available = args.cuda_available

    assert head_idx in [1,2,3,4]

    model = build_model()
    model_path = "../tdcn_models/bsds_checkpoint_epoch7.pth"
    model_load = torch.load(model_path)
    model.load_state_dict(model_load['state_dict'])
    model.eval()
    image2tentor = build_input(filename)

    if cuda_available:
        model = model.cuda()
        image2tentor = image2tentor.cuda()

    if fusion_mode:
        output = model(image2tentor)[-1][0, 0].detach()
        torchvision.utils.save_image(output, "%s_predF.png" % (filename.split('.')[0]))
    else:
        output = model(image2tentor)[head_idx-1][0, 0].detach()
        torchvision.utils.save_image(output, "%s_pred%s.png" % (filename.split('.')[0],str(head_idx)))

if __name__ == '__main__':
    main()
