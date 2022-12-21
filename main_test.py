import os
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from option import args
from data import HSRData
from models.network_eunet import EUNet
from utils import utils_model
from utils import utils_image as util


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('===> Loading testset')
    test_path = args.dir_data + args.dataset_name + '/test/X' + str(args.scale)
    test_set = HSRData(data_dir=test_path, sigma=args.sigma, augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print('===> Loading model')
    assert os.path.exists(args.model_path), 'Error: model_path is empty.'
    print(f'loading model from {args.model_path}')

    model = define_model(args)
    model = model.to(device)
    model.eval()

    if not os.path.exists('result/'):
        os.makedirs('result/')

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['sam'] = []

    for i, test_data in enumerate(test_loader):

        L = test_data['L'].to(device)
        S = test_data['S'].to(device)
        H = test_data['H'].to(device)

        E = model(L, S)

        E, H = E.squeeze().detach().cpu().numpy().transpose(1, 2, 0), H.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        E = np.clip(E, 0., 1.)

        p = util.calculate_psnr_hsi(H, E)
        s = util.calculate_ssim_hsi(H, E)
        sa = util.calculate_sam_hsi(H, E)

        print('{:->4d} | {:<4.2f}dB {:<4.4f} {:<4.2f}'.format(i, p, s, sa))

        test_results['psnr'].append(p)
        test_results['ssim'].append(s)
        test_results['sam'].append(sa)

    avg_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    avg_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    avg_sam = sum(test_results['sam']) / len(test_results['sam'])

    # testing log
    print('<Average PSNR : {:<.2f}dB, Average SSIM : {:<.4f}, Average SAM : {:<.2f}\n'.format(avg_psnr, avg_ssim, avg_sam))



def define_model(args):
    model = EUNet(scale=args.scale, n_iter=args.n_iters, n_colors=args.n_colors, n_feats=args.n_feats,
                  n_modules=args.n_modules, block=args.block_type, n_blocks=args.n_blocks, dilations=args.dilations,
                  expand_ratio=args.expand_ratio, is_blur=args.is_blur)

    param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)

    return model


if __name__ == '__main__':
    main()