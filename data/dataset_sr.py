import os
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
import utils.utils_image as util


class HSRData(data.Dataset):
    def __init__(self, data_dir, sigma=0., augment=None):
        super(HSRData, self).__init__()
        self.files = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        self.sigma = sigma
        self.augment = augment

    def __getitem__(self, index):
        aug_num = 0
        if self.augment:
            aug_num = np.random.randint(0, 8)

        data = sio.loadmat(self.files[index])
        lq = np.array(data['lq'][...], dtype=np.float32)
        it = np.array(data['it'][...], dtype=np.float32)
        gt = np.array(data['gt'][...], dtype=np.float32)
        lq, it, gt = util.augment_img(lq, mode=aug_num), util.augment_img(it, mode=aug_num), util.augment_img(gt, mode=aug_num)

        lq = torch.from_numpy(lq.copy()).permute(2, 0, 1)
        it = torch.from_numpy(it.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        sigma = np.array([self.sigma], dtype=np.float32)[:, None, None]
        sigma = torch.from_numpy(sigma.copy())

        return {'L': lq, 'I': it, 'H': gt, 'S': sigma}

    def __len__(self):
        return len(self.files)
