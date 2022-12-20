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

        # (1, 16)
        if self.iter == 0:
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.batnorm2d1 = nn.BatchNorm2d(self.out_channels)

            self.in_channels = self.out_channels
            # (16, 16) stride = 1
            self.conv2 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)

            # (16, 16) stride = 1 no relu
            self.conv3 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.batnorm2d3 = nn.BatchNorm2d(self.out_channels)

            self.out_channels = self.in_channels * 4
            self.conv4 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.batnorm2d4 = nn.Conv2d(self.out_channels)

            self.conv5 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
        else:
            self.batnorm2d1_1 = nn.BatchNorm2d(self.out_channels)
            self.strides = 2
            self.conv1_1 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.batnorm2d2_1 = nn.BatchNorm2d(self.out_channels)
            self.strides = 1
            self.conv2_1 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.batnorm2d3_1 = nn.BatchNorm2d(self.out_channels)
            self.strides = 1
            self.conv3_1 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)
            self.strides = 2
            self.out_channels = self.in_channels * 2
            self.conv4_1 = nn.Conv2d(self.in_channels, self.out_channels,
                                   self.kernel_size, self.strides, self.padding, self.dilation)

    def forward(self, x):
        if self.iter == 0:
            x = self.conv1(x)
            x = self.batnorm2d1(x)
            x = F.relu(x)
            y = self.conv2(x)
            y = self.conv3(y)
            y = self.batnorm2d3(y)
            y = F.relu(y)
            y = self.conv4(y)
            y = self.batnorm2d4(y)
            y = F.relu(y)
            x = x + y
            x = F.relu(x)
        else:
            y = self.batnorm2d1_1(x)
            y = F.relu(y)
            y = self.conv1_1(y)
            y = self.batnorm2d2_1(y)
            y = F.relu(y)
            y = self.conv2_1(y)
            y = self.batnorm2d3_1(y)
            y = F.relu(y)
            y = self.conv3_1(y)
            x = self.conv4_1(x)
            x = F.relu(x)
            x = x + y
        return x

class ResnetLayerV2Iter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1, dilation=1):
        super(ResnetLayerV2Iter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation = dilation

        # (16, 16) stride = 1
        self.batnorm2d1 = nn.BatchNorm2d(self.out_channels)
        self.out_channels = in_channels
        self.in_channels = out_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                               self.kernel_size, self.strides, self.padding, self.dilation)

        # (16, 16) stride = 1 no relu
        self.batnorm2d2 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels,
                                self.kernel_size, self.strides, self.padding, self.dilation)

        self.batnorm2d3 = nn.BatchNorm2d(self.out_channels)
        self.out_channels = out_channels
        self.in_channels = in_channels
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
