import numpy as np
import torch
from net.tdcn_deploy import TDCN
import torch.optim as optim
from net.data_loader import gt2attengt, Signle_Loader
from torch.utils.data import DataLoader
import torch.nn as nn

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

# for BSDS
def main():

    # dirs
    output_dir = 'lmc_deployment'
    dataset_dir = 'D:\lmc\TDCN-dataset'
    test_dataset = Signle_Loader(root=dataset_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, drop_last=True, shuffle=False)

    model = build_model()
    model_path = "../tdcn_models/bsds_checkpoint_epoch7.pth"
    model_load = torch.load(model_path)
    model.load_state_dict(model_load['state_dict'])

    for idx, (image, filename) in enumerate(test_loader):
        example_input = image
        break

    model.eval()
    torch.save(model.state_dict(), output_dir+'/TDCN_raw.pth')
    #scriptedm = torch.jit.script(model)
    #torch.jit.save(scriptedm, "TDCN_deploy.pt")
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, output_dir+"/TDCN_deploy.pt")

if __name__ == '__main__':
    main()
