import torch
import torch.nn as nn
import os
import math
import imgproc
from torch.nn.parameter import Parameter
import torch.nn.functional as function
from attention import SEWeightModule, SELayer, eca_layer, EAB, CBAMLayer, ESA
import torch.nn.functional as F

__all__ = ["default_conv", "conv", "conv1x1", "get_pyconv", "ChannelAttentionLayer",
           "NewPyconvRcan"]


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class PyConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3] // 2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class PyConv3(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)


class PyConv2(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)


class SoftPool2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(SoftPool2D, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class ResLayerPool(nn.Module):
    """3X3卷积/2+ 2X2pooling残差/2"""

    def __init__(self, inchannel, outchannel):
        super(ResLayerPool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=2, bias=False),

        )
        self.pool = SoftPool2D(2, 2)
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        res = self.pool(x)

        return out + res


class NewPyconvRcan(nn.Module):
    def __init__(self, sensing_rate):
        super(NewPyconvRcan, self).__init__()

        self.sensing_rate = sensing_rate
        self.base = 64
        self.blocksize = 32
        n_feats_csm = 32
        n_feats_in = 64

        if sensing_rate == 0.5:
            # 0.5000
            self.csm = nn.Sequential(
                nn.Conv2d(1, n_feats_in, kernel_size=3, padding=1, stride=1, bias=False),
                PyConv2d(in_channels=n_feats_in, out_channels=[n_feats_in//4, n_feats_in//4], pyconv_kernels=[3, 5], pyconv_groups=[1, 4]),
                ResLayerPool(n_feats_csm, n_feats_csm),
                nn.Conv2d(n_feats_csm, 2, kernel_size=1, padding=0, stride=1, bias=False),
            )

            self.initial = nn.Conv2d(2, 4, kernel_size=1, padding=0, stride=1, bias=False)
            self.measurement = 2
            self.m = 2
            self.m1 = 2
        elif sensing_rate == 0.25:
            # 0.2500
            self.csm = nn.Sequential(
                nn.Conv2d(1, n_feats_in, kernel_size=3, padding=1, stride=1, bias=False),
                PyConv2d(in_channels=n_feats_in, out_channels=[n_feats_in//4, n_feats_in//4], pyconv_kernels=[3, 5], pyconv_groups=[1, 4]),
                ResLayerPool(n_feats_csm, n_feats_csm),
                ResLayerPool(n_feats_csm, n_feats_csm),
                nn.Conv2d(n_feats_csm, 4, kernel_size=1, padding=0, stride=1, bias=False)
            )
            self.measurement = 4
            self.initial = nn.Conv2d(4, 16, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 2
            self.m1 = 4
        elif sensing_rate == 0.125:
            # 0.1250
            self.csm = nn.Sequential(
                nn.Conv2d(1, n_feats_in, kernel_size=3, padding=1, stride=1, bias=False),
                PyConv2d(in_channels=n_feats_in, out_channels=[n_feats_in//4, n_feats_in//4], pyconv_kernels=[3, 5], pyconv_groups=[1, 4]),
                ResLayerPool(n_feats_csm, n_feats_csm),
                ResLayerPool(n_feats_csm, n_feats_csm),
                nn.Conv2d(n_feats_csm, 2, kernel_size=1, padding=0, stride=1, bias=False)
            )
            self.measurement = 2
            self.initial = nn.Conv2d(2, 16, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 2
            self.m1 = 4
        elif sensing_rate == 0.0625:
            # 0.0625
            self.csm = nn.Sequential(
                nn.Conv2d(1, n_feats_in, kernel_size=3, padding=1, stride=1, bias=False),
                PyConv2d(in_channels=n_feats_in, out_channels=[n_feats_in//4, n_feats_in//4], pyconv_kernels=[3, 5], pyconv_groups=[1, 4]),
                ResLayerPool(n_feats_csm, n_feats_csm),
                ResLayerPool(n_feats_csm, n_feats_csm),
                ResLayerPool(n_feats_csm, n_feats_csm),
                nn.Conv2d(n_feats_csm, 4, kernel_size=1, padding=0, stride=1, bias=False),

            )
            self.measurement = 4
            self.initial = nn.Conv2d(4, 64, kernel_size=1, padding=0, stride=1, bias=False)
            self.m1 = 8
            self.m = 2
        elif sensing_rate == 0.03125:
            # 0.03125

            self.csm = nn.Sequential(
                nn.Conv2d(1, n_feats_in, kernel_size=3, padding=1, stride=1, bias=False),
                PyConv2d(in_channels=n_feats_in, out_channels=[n_feats_in//4, n_feats_in//4], pyconv_kernels=[3, 5], pyconv_groups=[1, 4]),
                ResLayerPool(n_feats_csm, n_feats_csm),
                ResLayerPool(n_feats_csm, n_feats_csm),
                ResLayerPool(n_feats_csm, n_feats_csm),
                nn.Conv2d(n_feats_csm, 2, kernel_size=1, padding=0, stride=1, bias=False),
            )
            self.initial = nn.Conv2d(2, 64, kernel_size=1, padding=0, stride=1, bias=False)
            self.m1 = 8
            self.m = 2
        elif sensing_rate == 0.015625:
            # 0.015625

            self.csm = nn.Sequential(
                nn.Conv2d(1, n_feats_csm, kernel_size=3, padding=1, stride=1, bias=False),
                #PyConv2d(in_channels=n_feats_in, out_channels=[n_feats_in//4, n_feats_in//4], pyconv_kernels=[3, 5], pyconv_groups=[1, 4]),
                ResLayerPool(n_feats_csm, n_feats_csm),
                ResLayerPool(n_feats_csm, n_feats_csm),
                ResLayerPool(n_feats_csm, n_feats_csm),
                ResLayerPool(n_feats_csm, n_feats_csm),

                nn.Conv2d(n_feats_csm, 4, kernel_size=1, padding=0, stride=1, bias=False),
            )
            self.initial = nn.Conv2d(4, 256, kernel_size=1, padding=0, stride=1, bias=False)
            self.m1 = 16
            self.m = 2


        # First layer
        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, stride=1, bias=True)

        self.head = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=1, bias=True)
                                  

        # Residual Group
        trunk = []
        for _ in range(4):
            trunk.append(MultipleResidualsGroup(64))
        self.trunk = nn.Sequential(*trunk)

        # Second layer
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Upsampling layers
        upsampling = []
        if self.m == 2 or self.m == 4 or self.m == 8 or self.m == 16:
            for _ in range(int(math.log(self.m, 2))):
                upsampling.append(UpsampleBlock(64, 2))
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1))

        self.register_buffer("mean", torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        xsample = self.csm(x)
        x = self.initial(xsample)
        initial = nn.PixelShuffle(self.m1)(x)
        #print(initial.size())

        out = self.conv1(initial)
        out1 = self.head(out)
        #print(out1.size())
        out = self.relu(out1)

        out = self.trunk(out)
        out = self.conv2(out)
        out = torch.add(out, out1)

        #out = self.upsampling(out)
        a = self.conv3(out)

        return a, initial


class ChannelAttentionLayer(nn.Module):
    """Attention Mechanism Module"""

    def __init__(self, channel: int, reduction=16):
        super(ChannelAttentionLayer, self).__init__()
        self.channel_attention_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, (1, 1), (1, 1), (0, 0)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = x
        out = self.channel_attention_layer(x)
        out = torch.mul(out, identity)
        out = torch.add(out, y)
        return out


'''MCAB前期特征提取部分'''


class splitblock(nn.Module):
    def __init__(self, inplanes, outplanes, scales=4):
        super(splitblock, self).__init__()
        self.act = activation('lrelu', neg_slope=0.05)

        if outplanes % scales != 0:  # 输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')

        self.scales = scales
        # 1*1的卷积层
        self.inconv = nn.Sequential(
            nn.Conv2d(inplanes, 64, 1, 1, 0),
        )
        # 3*3 5*5 7*7 9*9 的卷积层，一共有4个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 5, 1, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 7, 1, 3),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, 9, 1, 4),
        )
        # 1*1的卷积层
        self.outconv = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
        )

    def forward(self, x):
        input = x
        x = self.inconv(x)

        # scales个部分
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(self.conv1(xs[0]))
        ys.append(self.act(self.conv2(xs[1] + ys[0])))
        ys.append(self.act(self.conv3(xs[2] + self.conv2(xs[1] + ys[0]))))
        ys.append(self.act(self.conv4(xs[3] + self.conv3(xs[2] + self.conv2(xs[1] + ys[0])))))
        y = torch.cat(ys, 1)
        y = self.outconv(y)
        output = y + input
        return output


class MCAB(nn.Module):
    """ MCAB"""

    def __init__(self, channel: int):
        super(MCAB, self).__init__()
        self.mcab = nn.Sequential(
            splitblock(channel, channel, 4),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)),
        )

        self.ca = nn.Sequential(
            ChannelAttentionLayer(channel),
            nn.Conv2d(channel, channel, 1, 1, 0),
        )
        self.sa = ESA(channel, nn.Conv2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.mcab(x)

        residual = out
        out = self.ca(out)
        out = self.sa(out)
        out = torch.add(out, residual)

        out = torch.add(out, identity)
        return out


class MultipleResidualsGroup(nn.Module):
    """MultipleResidualsGroup (MG) 自定义权重版"""

    def __init__(self, channel: int):
        super(MultipleResidualsGroup, self).__init__()

        # Redidual dense blocks
        self.MGs = nn.ModuleList()
        for _ in range(4):
            self.MGs.append(MCAB(channel))

        # Global Feature Fusion
        self.IFF = nn.Sequential(*[
            nn.Conv2d(4 * channel, channel, 1, padding=0, stride=1),
            # nn.Conv2d(channel, channel, 1, padding=0, stride=1),
            nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1))
        ])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        MGs_out = []
        for i in range(4):
            x = self.MGs[i](x)
            MGs_out.append(x)

        x_out = torch.cat(MGs_out, 1)
        out = self.IFF(x_out)

        # 残差
        out = torch.add(out, identity)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, m: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.m = m
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * self.m * self.m, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(m),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)

        return out


def activation(act_type, inplace=False, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer




