import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from functools import reduce
from operator import __add__

class ResnetBlockLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(ResnetBlockLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                                self.kernel_size, self.strides, self.padding)

        self.batnorm2d = nn.BatchNorm2d(self.out_channels)


    def forward(self, x, activation='relu', batch_normalization=True, conv_first=True):
        if conv_first:
            x = self.conv(x)
            if batch_normalization:
                x = self.batnorm2d(x)
            if activation is not None:
                x = F.relu(x)
        else:
            if batch_normalization:
                x = self.batnorm2d(x)
            if activation is not None:
                x = F.relu(x)
            x = self.conv(x)
        return x

