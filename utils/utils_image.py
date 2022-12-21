import os
import cv2
import math
import torch
import random
import numpy as np
import scipy.io as sio
from datetime import datetime
from sklearn.metrics import mean_squared_error


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.mat']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


'''
# --------------------------------------------
# get image pathes
# --------------------------------------------
'''


# support multiple datasets
def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


'''
# --------------------------------------------
# makedir
# --------------------------------------------
'''


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


# Replace with your python file
def write_python_file(pypath, path):
    with open(pypath) as f:
        data = f.read()

    with open(path, mode="w") as f:
        f.write(data)


# --------------------------------------------
# get single mat of size HxWxn_channles (>3)
# --------------------------------------------
def imread_mat(path, key_str):
    # input: path
    # output: HxWxn_channels
    img = np.array(sio.loadmat(path)[key_str][...], dtype=np.float32)
    return img


def imread_multimat(path, key_strs):
    # input: path, key_strs: [list]
    # output: HxWxn_channels
    data = sio.loadmat(path)
    imgs = []
    for key_str in key_strs:
        imgs.append(np.array(data[key_str][...], dtype=np.float32))
    return imgs


'''
# --------------------------------------------
# Augmentation, flip and/or rotate
# --------------------------------------------
'''


def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


'''
# --------------------------------------------
# modcrop and shave
# --------------------------------------------
'''


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def shave(img_in, border=0):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    h, w = img.shape[:2]
    img = img[border:h-border, border:w-border]
    return img


# --------------------------------------------
# numpy(single) (HxWxC) <--->  tensor
# --------------------------------------------


# convert single (HxWxC) to 3-dimensional torch tensor
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


# convert single (HxWxC) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


# convert torch tensor to single
def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img

# convert torch tensor to single
def tensor2single3(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


def single2tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float().unsqueeze(0)


def single32tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).unsqueeze(0)


def single42tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float()


'''
# --------------------------------------------
# HSI metric, PSNR, SSIM and SAM
# --------------------------------------------
'''


# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr_hsi(x_true, x_pre):
    '''calculate PSNR
    img1, img2: [0, 1]
    '''
    if not x_true.shape == x_pre.shape:
        raise ValueError('Input images must have the same dimensions.')
    n_bands = x_true.shape[2]
    PSNR = np.zeros(n_bands)
    MSE = np.zeros(n_bands)
    mask = np.ones(n_bands)

    x_true, x_pre = x_true.astype(np.float32), x_pre.astype(np.float32)

    for k in range(n_bands):
        x_true_k = x_true[:, :, k].reshape([-1])
        x_pre_k = x_pre[:, :, k, ].reshape([-1])
        MSE[k] = mean_squared_error(x_true_k, x_pre_k)  # ==> compare_mse
        MAX_k = np.max(x_true_k)
        if MAX_k != 0:
            PSNR[k] = 10 * math.log10(math.pow(MAX_k, 2) / MSE[k])
            # PSNR[k] = 10 * math.log10(math.pow(1, 2) / MSE[k]) ==> compare_psnr
        else:
            mask[k] = 0

    psnr = PSNR.sum() / mask.sum()
    # mse = MSE.mean()
    # return psnr, mse
    return psnr


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim_hsi(x_true, x_pre):
    '''calculate SSIM
    img1, img2: [0, 1]
    '''
    if not x_true.shape == x_pre.shape:
        raise ValueError('Input images must have the same dimensions.')
    x_true, x_pre = x_true.astype(np.float32), x_pre.astype(np.float32)
    num = x_true.shape[2]
    ssimm = np.zeros(num)
    c1 = (0.01 * 1) ** 2
    c2 = (0.03 * 1) ** 2
    n = 0
    for x in range(x_true.shape[2]):
        z = np.reshape(x_pre[:, :, x], [-1])
        sa = np.reshape(x_true[:, :, x], [-1])
        y = [z, sa]
        cov = np.cov(y)
        oz = cov[0, 0]
        osa = cov[1, 1]
        ozsa = cov[0, 1]
        ez = np.mean(z)
        esa = np.mean(sa)
        ssimm[n] = ((2*ez*esa+c1) * (2*ozsa+c2)) / ((ez*ez+esa*esa+c1) * (oz+osa+c2))
        n = n + 1
    SSIM = np.mean(ssimm)
    return SSIM


# --------------------------------------------
# SAM
# --------------------------------------------
def calculate_sam_hsi(x_true, x_pre):
    '''calculate SAM
    img1, img2: [0, 1]
    '''
    if not x_true.shape == x_pre.shape:
        raise ValueError('Input images must have the same dimensions.')
    x_true, x_pre = x_true.astype(np.float32), x_pre.astype(np.float32)
    num = (x_true.shape[0]) * (x_true.shape[1])
    samm = np.zeros(num)
    n = 0
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            z = np.reshape(x_pre[x, y, :], [-1])
            sa = np.reshape(x_true[x, y, :], [-1])
            tem1 = np.dot(z, sa)
            tem2 = (np.linalg.norm(z)) * (np.linalg.norm(sa))
            samm[n] = np.arccos(tem1 / tem2) / np.pi * 180.
            n = n + 1
    idx = (np.isfinite(samm)) # Array of bool
    SAM = np.sum(samm[idx]) / np.sum(idx)
    if np.sum(~idx) != 0:
        print("waring: some values were ignored when computing SAM")
    return SAM