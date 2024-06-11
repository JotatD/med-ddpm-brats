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
inputfolder = "/home/juan.lugo/Code/med-ddpm/fast_sampling_input"
img_dir = "../fast_sampling_results/image"
msk_dir = "../fast_sampling_results/mask"

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

from utils_brats import label2masks, resize_img_4d, input_transform, processImg, processMsk


input_size = 192
depth_size = 144


for sampling_step in range(250, 251, 10):
    exportfolder = f"/home/juan.lugo/Code/med-ddpm/fast_sampling_results/image_{sampling_step}"
    os.makedirs(exportfolder, exist_ok=True)

    os.makedirs(f"{exportfolder}/t1", exist_ok=True)
    os.makedirs(f"{exportfolder}/t1ce", exist_ok=True)
    os.makedirs(f"{exportfolder}/t2", exist_ok=True)
    os.makedirs(f"{exportfolder}/flair", exist_ok=True)
    os.makedirs(f"{exportfolder}/seg", exist_ok=True)

    
    for counter, mask in enumerate(mask_list):
        name = mask.split("/")[-1].split(".")[0]
        img = nib.load(mask).get_fdata()
        img = label2masks(img)
        img = resize_img_4d(img)
        input_tensor = input_transform(img)
        diffusion = IU.make_diffusion("./weights/3dcddpm_params.pth", 250, sampling_step)
        wrap = IU.Wrap(model, input_tensor.to(device)).to(device)
        with torch.no_grad():
            vis = diffusion.p_sample_loop(wrap, (1, 4, 144, 192, 192), progress=True)
            t1_images = vis[:, 0, ...]
            t1ce_images = vis[:, 1, ...]
            t2_images = vis[:, 2, ...]
            flair_images = vis[:, 3, ...]

            t1_images = t1_images.transpose(3, 1)
            t1ce_images = t1ce_images.transpose(3, 1)
            t2_images = t2_images.transpose(3, 1)
            flair_images = flair_images.transpose(3, 1)

            ref = nib.load(mask)
            b = 0
            sampleImage = t1_images[b, :, :, :].cpu().numpy()
            sampleImage=sampleImage.reshape([input_size, input_size, depth_size])
            nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
            nib.save(nifti_img, f"{exportfolder}/t1/{counter}_{name}_t1.nii.gz")
            processImg(f"{exportfolder}/t1/{counter}_{name}_t1.nii.gz")
            
            sampleImage = t1ce_images[b, :, :, :].cpu().numpy()
            sampleImage=sampleImage.reshape([input_size, input_size, depth_size])
            nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
            nib.save(nifti_img, f"{exportfolder}/t1ce/{counter}_{name}_t1ce.nii.gz")
            processImg(f"{exportfolder}/t1ce/{counter}_{name}_t1ce.nii.gz")
            
            sampleImage = t2_images[b, :, :, :].cpu().numpy()
            sampleImage=sampleImage.reshape([input_size, input_size, depth_size])
            nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
            nib.save(nifti_img, f"{exportfolder}/t2/{counter}_{name}_t2.nii.gz")
            processImg(f"{exportfolder}/t2/{counter}_{name}_t2.nii.gz")
            
            sampleImage = flair_images[b, :, :, :].cpu().numpy()
            sampleImage=sampleImage.reshape([input_size, input_size, depth_size])
            nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
            nib.save(nifti_img, f"{exportfolder}/flair/{counter}_{name}_flair.nii.gz")
            processImg(f"{exportfolder}/flair/{counter}_{name}_flair.nii.gz")
            
            mask = ref.get_fdata()
            mask[mask==4.] = 0.
            mask[mask==2.] = 5.
            mask[mask==1.] = 2.
            mask[mask==5.] = 1.
            nifti_img = nib.Nifti1Image(mask, affine=ref.affine)
            nib.save(nifti_img, f"{exportfolder}/seg/{counter}_{name}_seg.nii.gz")
            processMsk(f"{exportfolder}/seg/{counter}_{name}_seg.nii.gz")
            
            torch.cuda.empty_cache()

    print("OK!")
        
