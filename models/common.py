import math
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1, dilation=1):
    '''
        padding corresponding kernel 3
    '''
    if dilation == 1:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), bias=bias, groups=groups)
    elif dilation == 2:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=2, bias=bias, dilation=dilation, groups=groups)
    elif dilation == 3:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=3, bias=bias, dilation=dilation, groups=groups)
    elif dilation == 5:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=5, bias=bias, dilation=dilation, groups=groups)
    elif dilation == 9:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=9, bias=bias, dilation=dilation, groups=groups)
    else:
        print('unsupported dilation/kernel')
        return


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True, group=1, dilation=1,
            bn=False, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias, dilation=dilation)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ESALayer(nn.Module):
    """Constructs an Efficient Spectral Attention module.
    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(ESALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)    # N,C,1,1
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # N,C,1-->N,1,C-->N,C,1-->N,C,1,1
        # residual conncetion
        y = 1 + self.sigmoid(y)

        return x * y.expand_as(x)


class ESSB(nn.Module):
    def __init__(self, conv, n_feats, kernel_size=3, expand_ratio=2, dilation=1, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, attn=None):
        super(ESSB, self).__init__()
        hidden_dim = n_feats * expand_ratio

        if expand_ratio == 1:
            # dw
            m = [conv(hidden_dim, hidden_dim, kernel_size, bias=bias, dilation=dilation, groups=hidden_dim)]
            if bn:
                m.append(nn.BatchNorm2d(hidden_dim))
            m.append(act)
            # pw
            m.append(conv(hidden_dim, hidden_dim, 1, bias=bias))
        else:
            # pw
            m = [conv(n_feats, hidden_dim, 1, bias=bias)]
            if bn:
                m.append(nn.BatchNorm2d(hidden_dim))
            m.append(act)
            # dw
            m.append(conv(hidden_dim, hidden_dim, kernel_size, bias=bias, dilation=dilation, groups=hidden_dim))
            if bn:
                m.append(nn.BatchNorm2d(hidden_dim))
            m.append(act)
            # pw-linear
            m.append(conv(hidden_dim, n_feats, 1, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(hidden_dim))

        if attn is not None:
            m.append(attn)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ESSG(nn.Module):
    def __init__(self, block, conv, n_feats, kernel_size, n_blocks, dilations, expand_ratio, bias=True, bn=False, act=nn.ReLU(True), res=True, res_scale=1, attn=None, attn_block=None):
        super(ESSG, self).__init__()
        assert len(dilations) == n_blocks
        m = [block(conv, n_feats, kernel_size, dilation=dilations[i], expand_ratio=expand_ratio, bias=bias, bn=bn, act=act, res_scale=res_scale, attn=attn) for i in range(n_blocks)]
        if attn_block is not None:
            m.append(attn)
        self.body = nn.Sequential(*m)
        self.res = res
        self.res_scale = res_scale

    def forward(self, x):
        if self.res:
            res = self.body(x).mul(self.res_scale)
            res += x
        else:
            res = self.body(x)

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, method='deconv', bn=False, act=None, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                if method == 'deconv':
                    m.append(nn.ConvTranspose2d(n_feats, n_feats, kernel_size=2, stride=2, padding=0, bias=bias))
                elif method == 'espcn':
                    m.append(conv(n_feats, 4 * n_feats, 3, bias))
                    m.append(nn.PixelShuffle(2))
                elif method == 'idw':
                    m.append(nn.Upsample(scale_factor=2))
                    m.append(conv(n_feats, n_feats, 3, bias))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act is not None:
                    m.append(act)

        elif scale == 3:
            if method == 'deconv':
                m.append(nn.ConvTranspose2d(n_feats, n_feats, kernel_size=3, stride=3, padding=0, bias=bias))
            elif method == 'espcn':
                m.append(conv(n_feats, 9 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(3))
            elif method == 'idw':
                m.append(nn.Upsample(scale_factor=3))
                m.append(conv(n_feats, n_feats, 3, bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act is not None:
                m.append(act)
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# if __name__ == '__main__':
#     b = ESSG(block=ESSB, conv=default_conv, n_feats=64, kernel_size=3, n_blocks=2, dilations=[1,2],
#          expand_ratio=2, bias=False, act=None, res=False, attn=ESALayer(k_size=5))
#     print(b)

