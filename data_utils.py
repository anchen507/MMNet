import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import os
import math
# import cv2


def random_crop(HR, patch_size): # HR: N*H*W
    _, _, h_hr, w_hr = HR.shape

    h_start_hr = random.randint(5, h_hr - patch_size - 5)
    h_end_hr = h_start_hr + patch_size
    w_start_hr = random.randint(5, w_hr - patch_size - 5)
    w_end_hr = w_start_hr + patch_size

    HR = HR[:, :, h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    return HR

def add_noise(img, n_std):
    return img + np.random.normal(0, n_std, img.shape)

def add_light(img, light, *paras, mode):
    if mode == 'point':
        x0, y0, radius = paras
        light_res = np.zeros(3, radius, radius)
        for i in range(radius):
            for j in range(radius):
                light_res[0, i, j, 0] = light * (1-math.sqrt((i-radius//2)**2 + (j-radius//2)**2)/radius)
                light_res[1, i, j, 0] = light * (1-math.sqrt((i-radius//2)**2 + (j-radius//2)**2)/radius)
                light_res[2, i, j, 0] = light * (1-math.sqrt((i-radius//2)**2 + (j-radius//2)**2)/radius)

        light_res = np.clip(light_res + img[:, x0-radius//2:x0+1+radius//2, y0-radius//2:y0+1+radius//2, :], 0, 255)
        img[:, x0-radius//2:x0+1+radius//2, y0-radius//2:y0+1+radius//2, :] = light_res
    return img

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''

    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def ycbcr2rgb(ycbcr_img):
    ycbcr_img = ycbcr_img.numpy()
    in_img_type = ycbcr_img.dtype
    if in_img_type != np.uint8:
        ycbcr_img *= 255.
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.array([16, 128, 128])
    rgb_img = np.zeros(ycbcr_img.shape)
    for x in range(ycbcr_img.shape[0]):
        for y in range(ycbcr_img.shape[1]):
            rgb_img[x, y, :] = np.maximum(0, np.minimum(255,np.round(np.dot(mat_inv, ycbcr_img[x, y, :] - offset) * 255.0)))
    return torch.from_numpy(np.ascontiguousarray(rgb_img.astype(np.float32)/255))
