import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from functools import reduce
from operator import __add__


class ResnetLayerV2(nn.Module):
    def __init__(self, iter, in_channels, out_channels, kernel_size=3, strides=1, padding=1, dilation=1):
        super(ResnetLayerV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.iter = iter

        if self.iter == 0:
            # (1, 16) (28, 28) stride 1 kernel_size = 3
            self.kernel_size = 3
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.batnorm2d1 = nn.BatchNorm2d(self.out_channels)

            # (16, 16) stride = 1 kernel_size = 1
            self.in_channels = self.out_channels
            self.kernel_size = 1
            self.padding = 0
            self.conv2 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.batnorm2d2 = nn.BatchNorm2d(self.out_channels)

            # (16, 16) stride = 1 kernel_size = 3
            self.kernel_size = 3
            self.padding = 1
            self.conv3 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.batnorm2d3 = nn.BatchNorm2d(self.out_channels)

            # (16, 64) stride = 1 kernel_size = 1
            self.out_channels = self.in_channels * 4
            self.kernel_size = 1
            self.padding = 0
            self.conv4 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.batnorm2d4 = nn.BatchNorm2d(self.out_channels)

            # (16, 64) stride = 1 kernel_size = 1
            self.kernel_size = 1
            self.padding = 0
            self.conv5 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.batnorm2d5 = nn.BatchNorm2d(self.out_channels)
        else:
            # (16, 64) (28, 28) stride = 2
            # (64, 128) (14, 14) stride = 2
            self.batnorm2d1 = nn.BatchNorm2d(self.out_channels)
            self.strides = 2
            self.kernel_size = 1
            self.padding = 0
            # (64, 64) (14, 14) stride = 2
            # (128, 128) (7, 7) stride = 2
            if iter == 1:
                self.in_channels = in_channels * 4
            else:
                self.in_channels = in_channels * 2
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.batnorm2d2 = nn.BatchNorm2d(self.out_channels)
            self.strides = 1
            self.kernel_size = 3
            self.padding = 1
            # (64, 64) (14, 14) stride = 1
            # (128, 128) (7, 7) stride = 1
            self.conv2 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.batnorm2d3 = nn.BatchNorm2d(self.out_channels)
            self.strides = 1
            self.kernel_size = 1
            self.padding = 0
            # (64, 128) (14, 14) stride = 1
            # (128, 256) (7, 7) stride = 1
            self.out_channels = self.out_channels * 2
            self.conv3 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.strides = 2
            self.kernel_size = 1
            self.padding = 0
            # (64, 128) (14, 14) stride = 2
            # (128, 256) (7, 7) stride = 2
            self.conv4 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)

    def forward(self, x):
        if self.iter == 0:
            x = self.conv1(x)
            x = self.batnorm2d1(x)
            x = F.relu(x)
            y = self.conv2(x)
            y = self.batnorm2d2(y)
            y = F.relu(y)
            y = self.conv3(y)
            y = self.batnorm2d3(y)
            y = F.relu(y)
            y = self.conv4(y)
            # y = self.batnorm2d4(y)
            # y = F.relu(y)
            x = self.conv5(x)
            x = x + y
        else:
            y = self.batnorm2d1(x)
            y = F.relu(y)
            y = self.conv1(y)
            y = self.batnorm2d2(y)
            y = F.relu(y)
            y = self.conv2(y)
            y = self.batnorm2d3(y)
            y = F.relu(y)
            y = self.conv3(y)
            x = self.conv4(x)
            x = F.relu(x)
            x = x + y
        return x

class ResnetLayerV2Iter(nn.Module):
    def __init__(self, iter, in_channels, out_channels, kernel_size=3, strides=1, padding=1, dilation=1):
        super(ResnetLayerV2Iter, self).__init__()
        self.iter = iter
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        # (16, 64)
        # (64, 128)
        self.batnorm2d1 = nn.BatchNorm2d(self.out_channels)
        # (64, 16) stride = 1 kernel_size = 1
        # (128, 64) stride = 1 kernel_size = 1
        self.out_channels = in_channels
        self.in_channels = out_channels
        self.kernel_size = 1
        self.padding = 0
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                               self.kernel_size, self.strides, self.padding, self.dilation)

        self.batnorm2d2 = nn.BatchNorm2d(self.out_channels)
        # (16, 16) stride = 1
        # (64, 64) stride = 1
        self.in_channels = in_channels
        self.kernel_size = 3
        self.padding = 1
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels,
                                self.kernel_size, self.strides, self.padding, self.dilation)

        self.batnorm2d3 = nn.BatchNorm2d(self.out_channels)
        # (16, 64) stride = 1
        # (64, 128) stride = 1
        if iter == 0:
            self.out_channels = in_channels * 4
        else:
            self.out_channels = in_channels * 2
        self.kernel_size = 1
        self.padding = 0
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels,
                                self.kernel_size, self.strides, self.padding, self.dilation)

    def forward(self, x):
        y = self.batnorm2d1(x)
        y = F.relu(y)
        y = self.conv1(y)
        y = self.batnorm2d2(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.batnorm2d3(y)
        y = F.relu(y)
        y = self.conv3(y)
        x = x + y
        return x
