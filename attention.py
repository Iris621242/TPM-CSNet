
import torch
import torch.nn as nn
import math
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import warnings
from torch.nn import init

__all__ = ["SELayer", "eca_layer", "SEWeightModule",]


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEWeightModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight


# class SCA(torch.nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = in_channels
#
#         self.maxpooling1 = nn.AdaptiveMaxPool2d(output_size=(16, 16))
#         self.avgpooling1 = nn.AdaptiveAvgPool2d(output_size=(16, 16))
#
#         self.conv1 = nn.Conv2d(self.in_channels*2, self.in_channels, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(self.in_channels)
#
#         self.conv2 = nn.Conv2d(self.in_channels*2, self.out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(self.out_channels)
#
#     def forward(self, input):
#         '''
#         input : B X in_channels X H X W
#
#           return
#             result : B X out_channels X H X W
#         '''
#         max_x = self.maxpooling1(input)
#         avg_x = self.avgpooling1(input)
#
#         x = torch.cat((max_x, avg_x), dim=1)
#         x = F.relu(self.bn1(self.conv1(x)))
#
#         result = torch.cat((input, x), dim=1)
#         result = F.relu(self.bn2(self.conv2(result)))
#
#         return result
#
#
# class CCA(torch.nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = in_channels
#
#         self.maxpooling1 = nn.AdaptiveMaxPool2d(output_size=(1, 1))
#         self.avgpooling1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#
#         self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
#         self.bn1 = nn.BatchNorm2d(self.in_channels)
#         self.conv2 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
#         self.bn2 = nn.BatchNorm2d(self.in_channels)
#
#         self.conv3 = nn.Conv2d(self.in_channels*2, self.out_channels, kernel_size=1)
#         self.bn3 = nn.BatchNorm2d(self.out_channels)
#
#     def forward(self, input):
#         '''
#           input : B X in_channels X H X W
#
#           return
#             result : B X out_channels X H X W
#         '''
#         max_x = self.maxpooling1(input)
#         avg_x = self.avgpooling1(input)
#
#         max_x = F.relu(self.bn1(self.conv1(max_x)))
#         avg_x = F.relu(self.bn2(self.conv2(avg_x)))
#
#         encode = torch.cat((max_x, avg_x), dim=1)
#         encode = F.relu(self.bn3(self.conv3(encode)))
#
#         result = torch.mul(input, encode)
#
#         return result


# # 并联EAB：
#
# class EAB(nn.Module):
#     def __init__(self, planes):
#         super(EAB, self).__init__()
#         self.ca = CCA(planes)  # planes是feature map的通道个数
#         self.sa = SCA(planes)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         # channel split
#         x_0, x_1 = x.chunk(2, dim=1)
#
#         x0 = self.ca(x_0)
#         x1 = self.sa(x_1)
#         out = torch.cat([x0, x1], dim=1)
#         out = out.reshape(b, -1, h, w)
#         # out = self.channel_shuffle(out, 2)
#         return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


##直接单独加入CBAM：
class cbam(nn.Module):
    def __init__(self, planes):
        super(cbam, self).__init__()
        self.ca = ChannelAttention(planes)  # planes是feature map的通道个数
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x  # 广播机制
        x = self.sa(x) * x  # 广播机制
        return x


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


"""PSA"""

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


# 基于规范化的注意力模块(NAM)
class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #

        return x


class Att(nn.Module):
    def __init__(self, channels, shape, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)


    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1


"""Enhanced Spatial Attention (ESA)"""


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return m.expand_as(x)

        # return x * m.expand_as(x)


""" ECA"""
class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv11 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv11(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ChannelAttentionLayer(nn.Module):
    """Attention Mechanism Module"""

    def __init__(self, channel: int, reduction: int):
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

        out = self.channel_attention_layer(x)

        # out = torch.mul(out, identity)

        return out



class EAB(nn.Module):
    def __init__(self, planes,):
        super(EAB, self).__init__()
        # self.ca = eca_layer(planes)  # planes是feature map的通道个数
        self.ca = ChannelAttentionLayer(planes, 16)  # planes是feature map的通道个数
        self.sa = ESA(planes, nn.Conv2d)

    def forward(self, x):
        b, c, _, _ = x.size()
        # channel split
        # x_0, x_1 = x.chunk(2, dim=1)

        # x0 = self.ca(x)
        # x1 = self.sa(x)
        # out = torch.cat([x0, x1], dim=1)
        # out = 1 + self.sigmoid(x0 * x1)

        residual = x
        out = x*self.ca(x)
        out = out*self.sa(out)
        return out+residual

        # return x * out.expand_as(x)

"""EAB  eca_layer+ESA"""
"""EAB  ChannelAttentionLayer+ESA"""


