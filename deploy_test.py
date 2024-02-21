import os.path

import numpy as np
import torch
from net.tdcn_deploy import TDCN
import torch.optim as optim
from net.data_loader import gt2attengt, Signle_Loader
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from os.path import join, isdir

# for BSDS
def main():

    # dirs
    output_dir = 'lmc_deployment'
    save_dir = output_dir + '/epoch-1-testing-record-view'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_dir = 'D:\lmc\TDCN-dataset'
    test_dataset = Signle_Loader(root=dataset_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, drop_last=True, shuffle=False)

    model = torch.jit.load(output_dir+"/TDCN_deploy.pt")
    #model = model.cuda()

    for idx, (image, filename) in enumerate(test_loader):
        #image = image.cuda()
        b, c, h, w = image.shape
        if h != 321:
            print('ignored size')
            continue
        output = model(image)[-1][0, 0].detach()
        torchvision.utils.save_image(output, join(save_dir, "%s.png" % filename))

if __name__ == '__main__':
    main()
