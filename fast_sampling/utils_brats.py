from enum import IntEnum, Enum
import numpy as np
import torchio as tio
from torchvision.transforms import Compose, Lambda
import torch
class LabelEnum(IntEnum):
    BACKGROUND = 0
    TUMORAREA1 = 1
    TUMORAREA2 = 2
    TUMORAREA3 = 3
    BRAINAREA = 4

def label2masks(masked_img):
    result_img = np.zeros(masked_img.shape + (4,))   # ( (H, W, D) + (2,)  =  (H, W, D, 2)  -> (B, 2, H, W, D))
    result_img[masked_img==LabelEnum.TUMORAREA1.value, 0] = 1
    result_img[masked_img==LabelEnum.TUMORAREA2.value, 1] = 1
    result_img[masked_img==LabelEnum.TUMORAREA3.value, 2] = 1
    result_img[masked_img==LabelEnum.BRAINAREA.value, 3] = 1
    return result_img

def resize_img_4d(input_img, input_size=192, depth_size=144):
    h, w, d, c = input_img.shape
    result_img = np.zeros((input_size, input_size, depth_size, 4))
    if h != input_size or w != input_size or d != depth_size:
        for ch in range(c):
            buff = input_img.copy()[..., ch]
            img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
            cop = tio.Resize((input_size, input_size, depth_size))
            img = np.asarray(cop(img))[0]
            result_img[..., ch] += img
        return result_img
    else:
        return input_img
    
input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.permute(3, 0, 1, 2)),
    Lambda(lambda t: t.unsqueeze(0)),
    Lambda(lambda t: t.transpose(4, 2))
])

def processImg(img):
    t1 = tio.ScalarImage(img)
    subject = tio.Subject(image = t1)
    transforms = tio.RescaleIntensity((0, 1)), tio.CropOrPad((240, 240, 155))  
    transform = tio.Compose(transforms)
    fixed = transform(subject)
    fixed.image.save(img)

def processMsk(msk):
    t1 = tio.LabelMap(msk)
    subject = tio.Subject(mask = t1)
    transform = tio.CropOrPad((240, 240, 155))   
    fixed = transform(subject)
    fixed.mask.save(msk)