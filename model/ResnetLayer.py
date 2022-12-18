import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from functools import reduce
from operator import __add__

# class Conv2dSamePadding(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super(Conv2dSamePadding, self).__init__(*args, **kwargs)
#         self.zero_pad_2d = nn.ZeroPad2d(
#             reduce(
#                 __add__,
#                 [
#                     (k // 2 + (k - 2 * (k // 2)) - 1, k // 2)
#                     for k in self.kernel_size[::-1]
#                 ],
#             )
#         )
#     def forward(self, input):
#         return self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)
#
# def _calc_same_pad(i: int, k: int, s: int, d: int):
#     return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)
#
# def conv2d_same(
#         x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
#         padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
#     ih, iw = x.size()[-2:]
#     kh, kw = weight.size()[-2:]
#     pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
#     pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
#     x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
#     return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
#
# class Conv2dSame(nn.Conv2d):
#     """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
#     """
#
#     # pylint: disable=unused-argument
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2dSame, self).__init__(
#             in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
#
#     def forward(self, x):
#         return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResnetLayer(nn.Module):
    def __init__(self, iter, in_channels, out_channels, kernel_size=3, strides=1, padding=1, dilation=1):
        super(ResnetLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.iter = iter

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                               self.kernel_size, self.strides, self.padding, self.dilation)
        self.batnorm2d1 = nn.BatchNorm2d(self.out_channels)

        self.in_channels = self.out_channels
        # (16, 16) stride = 1
        self.strides = 1
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels,
                               self.kernel_size, self.strides, self.padding, self.dilation)
        self.batnorm2d2 = nn.BatchNorm2d(self.out_channels)
        # (16, 16) stride = 1 no relu
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels,
                               self.kernel_size, self.strides, self.padding, self.dilation)
        self.batnorm2d3 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        if self.iter == 0:
            x = self.conv1(x)
            x = self.batnorm2d1(x)
            x = F.relu(x)
            y = self.conv2(x)
            y = self.batnorm2d2(y)
            y = self.conv3(y)
            y = self.batnorm2d3(y)
            x = x + y
            x = F.relu(x)
        else:
            x = self.conv1(x)
            x = self.batnorm2d1(x)
            x = F.relu(x)
            y = self.conv2(x)
            y = self.batnorm2d2(y)
            x = self.conv3(x)
            x = x + y
            x = F.relu(x)
        return x

class ResnetLayerIter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1, dilation=1):
        super(ResnetLayerIter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation = dilation

        # (16, 16) stride = 1
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                               self.kernel_size, self.strides, self.padding, self.dilation)
        self.batnorm2d1 = nn.BatchNorm2d(self.out_channels)
        # (16, 16) stride = 1 no relu
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels,
                                self.kernel_size, self.strides, self.padding, self.dilation)
        self.batnorm2d2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.batnorm2d1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.batnorm2d2(y)
        x = x + y
        x = F.relu(x)
        return y
