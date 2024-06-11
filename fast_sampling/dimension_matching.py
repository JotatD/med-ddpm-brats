import torch
import numpy as np
import glob
import inference_utils as IU
import nibabel as nib

# +
import os 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# set_cuda_params()

from guided_diffusion.unet_brats import create_model
from trainer_brats import GaussianDiffusion

def load_ddpm_model(weights_file, device):
    model = create_model(image_size=192, num_channels=64, num_res_blocks=2, in_channels=8, out_channels=4).to(device)
    diffusion = GaussianDiffusion(
        model,
        image_size = 192,
        depth_size = 144,
        timesteps = 250,   # number of steps
        loss_type = 'l1',    # L1 or L2
        with_condition=True,
        channels=8
    ).cuda()
    weight = torch.load(weights_file, map_location='cuda')
    diffusion.load_state_dict(weight['ema'])
    pure_net = diffusion.denoise_fn
    return pure_net

device = torch.device("cuda")

model = create_model(image_size=192, num_channels=64, num_res_blocks=2, in_channels=8, out_channels=4).to(device)

blocks = model.input_blocks[1:]
emb = torch.randn(4, 256).to(device)
h = torch.randn(1, 64, 144, 192, 192).to(device)
for module in blocks:
    h = module(h, emb)
    print(h.shape)
