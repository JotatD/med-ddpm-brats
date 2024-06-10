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

model = load_ddpm_model("../model/model_brats.pt", device)
inputfolder = "../current_images/masks"
img_dir = "../fast_sampling_results/image"
msk_dir = "../fast_sampling_results/ddim/mask"
sampling_step = 10
os.makedirs(img_dir, exist_ok=True)
os.makedirs(msk_dir, exist_ok=True)

mask_list = sorted(glob.glob(f"{inputfolder}/*.nii.gz"))
print(len(mask_list))

def process_mask(mask):
    final_mask = np.zeros(shape=(4, 144, 192, 192), dtype=np.float32)
    mask = np.transpose(mask, axes = (2, 1, 0))
    final_mask[0][mask==1] = 1
    final_mask[1][mask==2] = 1
    final_mask[2][mask==3] = 1
    final_mask[3][mask==4] = 1

    for i in range(4):
        print("i number channel", final_mask[i].sum())
    
    return final_mask

for mask in mask_list:
    name = mask
    input_mask = process_mask(nib.load(mask).get_fdata())
    input_tensor = torch.tensor(input_mask, dtype=torch.float32)
    diffusion = IU.make_diffusion("./weights/3dcddpm_params.pth", 250, sampling_step)
    wrap = IU.Wrap(model, input_tensor.to(device)).to(device)
    with torch.no_grad():
        vis = diffusion.p_sample_loop(wrap, (4, 144, 192, 192), progress=True)

        sampleImage = vis.cpu().numpy()[0]
        ref = nib.load(mask)
        refImg = ref.get_fdata()
        sampleImage=sampleImage.reshape(refImg.shape)
        nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
        nib.save(nifti_img, os.path.join(img_dir, f'{name}'))
        refImg = refImg.astype(np.int8)
        refImg[refImg==1.]=0
        refImg[refImg==2.]=1
        nifti_img = nib.Nifti1Image(refImg, affine=ref.affine)
        nib.save(nifti_img, os.path.join(msk_dir, f'{name}'))
