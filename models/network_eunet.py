import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import *


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


"""
# --------------------------------------------
# (1) Prior module
# act as a non-blind super resolver
# --------------------------------------------
"""


class HSRPN(nn.Module):
    def __init__(self, scale, n_colors, n_feats, n_modules, n_blocks=2, dilations=[1,2], expand_ratio=2, conv=default_conv, block=ESSB):
        super(HSRPN, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        # act = nn.LeakyReLU(0.2, True)
        self.input = BasicBlock(conv, n_colors+1, n_feats, kernel_size, act=None)
        layers = []
        for i in range(n_modules):
            layers.append(ESSG(block=block, conv=conv, n_feats=n_feats, kernel_size=kernel_size, n_blocks=n_blocks, dilations=dilations,
                                expand_ratio=expand_ratio, bias=False, act=act, res=False, attn=ESALayer(k_size=5)))

        self.conv = nn.Sequential(*layers)
        self.up = Upsampler(conv, scale, n_feats, method='deconv', act=act)
        self.output = BasicBlock(conv, n_feats, n_colors, kernel_size, act=None)

    def forward(self, x, lx):
        x = self.input(x)
        x0 = x
        x = self.conv(x)
        x = torch.add(x, x0)
        x = self.up(x)
        x = self.output(x)
        x = torch.add(x, lx)
        return x


"""
# --------------------------------------------
# (2) Data consistency module
# a single step of gradient descent
# --------------------------------------------
"""


class DataNet(nn.Module):
    def __init__(self, scale, blur, deblur):
        super(DataNet, self).__init__()
        self.scale = scale
        self.B = blur
        self.B_T = deblur

    def forward(self, z, dx, y, alpha, delta):
        B_z = (1 - delta * alpha) * z - delta * self.B_T(self.B(z))
        z = B_z + delta * self.B_T(y) + delta * alpha * dx
        return z


"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""


class HyPaNet(nn.Module):
    def __init__(self, n_in=2, n_out=18, n_hid=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(n_in, n_hid, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hid, n_hid, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hid, n_out, 1, padding=0, bias=True),
                nn.Softplus())

    def forward(self, sigma, sf):
        sf = torch.tensor(sf).type_as(sigma).expand_as(sigma)
        x = torch.cat((sigma, sf), dim=1)
        x = self.mlp(x) + 1e-6
        return x


"""
# --------------------------------------------
# EUNet <= SR Prior + Data consistency + Hyper-parameter
# --------------------------------------------
"""


class EUNet(nn.Module):
    def __init__(self, scale=2, n_iter=4, n_colors=102, n_feats=64, n_modules=3, n_blocks=2, dilations=[1,2],
                 expand_ratio=2, conv=default_conv, block=ESSB, is_blur=False):
        super(EUNet, self).__init__()
        self.scale = scale
        self.n = n_iter

        if is_blur:
            self.B = nn.Conv2d(n_colors, n_colors, kernel_size=3, stride=1, padding=1, groups=n_colors, bias=False)
            self.B_T = nn.ConvTranspose2d(n_colors, n_colors, kernel_size=3, stride=1, padding=1, groups=n_colors, bias=False)
        else:
            self.B = Identity()
            self.B_T = Identity()

        self.d = DataNet(self.scale, self.B, self.B_T)
        self.p = HSRPN(scale, n_colors, n_feats, n_modules, n_blocks, dilations, expand_ratio, conv, block)
        self.h = HyPaNet(n_in=2, n_out=self.n*3, n_hid=64)

    def forward(self, y, sigma):
        params = self.h(sigma, self.scale)
        h, w = y.shape[-2:]

        z = self.B_T(y)
        for i in range(self.n):
            alpha = params[:, i:i+1, :, :]
            delta = params[:, i+self.n:i+self.n+1, :, :]
            beta = params[:, i+self.n*2:i+self.n*2+1, :, :]
            lz = F.interpolate(z, scale_factor=self.scale, mode='bicubic')
            x = self.p(torch.cat((z, beta.repeat(1, 1, h, w)), dim=1), lz)
            dx = F.interpolate(x, scale_factor=1 / self.scale, mode='bicubic')
            z = self.d(z, dx, y, alpha, delta)

        return x


if __name__ == "__main__":
    model = EUNet(2, 4, 102, n_feats=128, n_modules=3, n_blocks=2, dilations=[1,2], expand_ratio=2)
    print(model)
