import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ResnetLayer import ResnetLayer, ResnetLayerIter

class ResNetTorch(nn.Module):

    def __init__(self, version, layer, layeriter, depth, num_classes):
        super(ResNetTorch, self).__init__()
        self.model_name = 'ResNetTorch'
        self.depth = depth
        self.num_classes = num_classes
        self.version = version
        self.in_channels = 1
        self.out_channels = 16
        self.strides = 1
        self.kernel_size = 3
        self.iter = 0

        if self.version == 1:
            # (1, 16) stride = 1
            num_res_blocks = int((depth - 2) / 6)
            self.layer1 = self._make_layer(layer, layeriter=None)
            self.in_channels = self.out_channels
            self.layer2 = self._make_layer(layer=None, layeriter=layeriter, num_res_blocks=num_res_blocks)
            # (16, 32) stride = 2
            self.iter = 1
            self.out_channels = self.out_channels * 2
            self.strides = 2
            self.layer3 = self._make_layer(layer, layeriter=None)
            self.in_channels = self.out_channels
            self.strides = 1
            self.layer4 = self._make_layer(layer=None, layeriter=layeriter, num_res_blocks=num_res_blocks)
            # (32, 64) stride = 2
            self.iter = 2
            self.out_channels = self.out_channels * 2
            self.strides = 2
            self.layer5 = self._make_layer(layer, layeriter=None)
            self.in_channels = self.out_channels
            self.strides = 1
            self.layer6 = self._make_layer(layer=None, layeriter=layeriter, num_res_blocks=num_res_blocks)

            self.avgpool2d = nn.AvgPool2d(kernel_size=4)
            self.flatten = nn.Flatten()
            # self.fc1 = nn.Linear(49*16, 32)
            self.fc1 = nn.Linear(64, 32)
            self.dropout = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(32, 10)
        else:
            # (1, 16) stride = 1
            num_res_blocks = int((depth - 2) / 6)
            self.layer1 = self._make_layer(layer, layeriter=None)
            # (16, 64) stride = 1
            self.in_channels = self.out_channels
            self.out_channels = self.in_channels * 4
            self.layer2 = self._make_layer(layer=None, layeriter=layeriter, num_res_blocks=num_res_blocks)

            # (16, 64) stride = 2
            self.iter = 1
            self.strides = 2
            self.layer3 = self._make_layer(layer, layeriter=None)

            # (64, 128) stride = 1
            self.in_channels = self.out_channels
            self.out_channels = self.in_channels * 2
            self.strides = 1
            self.layer4 = self._make_layer(layer=None, layeriter=layeriter, num_res_blocks=num_res_blocks)

            # (64, 128) stride = 2
            self.iter = 2
            self.strides = 2
            self.layer5 = self._make_layer(layer, layeriter=None)

            # (128, 256) stride = 1
            self.in_channels = self.out_channels
            self.out_channels = self.in_channels * 2
            self.stride = 1
            self.layer6 = self._make_layer(layer=None, layeriter=layeriter, num_res_blocks=num_res_blocks)

            self.batnorm2d = nn.BatchNorm2d(self.out_channels)
            self.avgpool2d = nn.AvgPool2d(kernel_size=4)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(256, 128)
            self.dropout = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

    def _make_layer(self, layer=None, layeriter=None, num_res_blocks=0):
        layers = []
        if layer is not None:
            layers.append(layer(self.iter, self.in_channels, self.out_channels, self.kernel_size, self.strides))
        elif layeriter is not None:
            for i in range(num_res_blocks):
                layers.append(layeriter(self.iter, self.in_channels, self.out_channels, self.kernel_size, self.strides))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.version == 1:
            self.ResnetV1(x)
        else:
            self.ResnetV2(x)
        return x

    def ResnetV1(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avgpool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def ResnetV2(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.batnorm2d(x)
        x = F.relu(x)
        x = self.avgpool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x