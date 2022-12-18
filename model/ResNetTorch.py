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

    def _make_layer(self, layer=None, layeriter=None, num_res_blocks=0):
        layers = []
        if layer is not None:
            layers.append(layer(self.iter, self.in_channels, self.out_channels, self.kernel_size, self.strides))
        elif layeriter is not None:
            for i in range(num_res_blocks):
                layers.append(layeriter(self.in_channels, self.out_channels, self.kernel_size, self.strides))
        return nn.Sequential(*layers)

    def forward(self, x):
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

    def ResnetV1(self, x):
        # if (depth - 2) % 6 != 0:
        #     raise ValueError('depth should be 6n+2 (eg 20, 32, in [a])')
        in_channels = 1
        out_channels = 16
        # num_res_blocks = int((depth - 2) / 6)

        x = self.ResnetLayer(x, in_channels=in_channels, out_channels=out_channels)
        #
        # # in_channels = out_channels
        # # for stack in range(3):
        # #     for res_block in range(num_res_blocks):
        # #         strides = 1
        # #         if stack > 0 and res_block == 0:
        # #             strides = 2
        # #         y = self.ResnetLayer(inputs=x,
        # #                              in_channels=in_channels,
        # #                              out_channels=out_channels,
        # #                              strides=strides)
        # #         y = self.ResnetLayer(inputs=y,
        # #                              in_channels=in_channels,
        # #                              out_channels=out_channels,
        # #                              activation=None)
        # #
        # #         if stack > 0 and res_block == 0:
        # #             x = self.ResnetLayer(inputs=x,
        # #                                  in_channels=in_channels,
        # #                                  num_filters=out_channels,
        # #                                  kernel_size=1,
        # #                                  strides=strides,
        # #                                  activation=None,
        # #                                  batch_normalization=False)
        # #             in_channels = out_channels
        # #         x = torch.cat([x, y], dim=1)
        # #         x = F.relu(x)
        # #     out_channels *= 2
        # #
        x = self.avgpool2d(x)
        y = self.flatten(x)
        y = self.linear1(y)
        y = F.relu(y)
        y = self.dropout(y)
        outputs = self.linear2(y)
        return outputs



"""
    def ResnetV2(self, x, depth, num_classes=10):
        if (depth -2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 110 in [b]')
        in_channels = 1
        out_channels = 16
        num_res_blocks = int((depth - 2) / 9)

        x = self.ResnetLayer(x,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        conv_first=True)
        in_channels = out_channels
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    out_channels = in_channels * 4
                    if res_block == 0:
                        activation = None
                        batch_normalization = False
                else:
                    out_channels = in_channels * 2
                    if res_block == 0:
                        strides = 2

                y = self.ResnetLayer(inputs=x,
                                     in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=activation,
                                     batch_normalization=batch_normalization,
                                     conv_first=False)
                y = self.ResnetLayer(inputs=y,
                                     in_channels=in_channels,
                                     out_channels=out_channels,
                                     conv_first=False)
                y = self.ResnetLayer(inputs=y,
                                     in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=1,
                                     conv_first=False)
                if res_block == 0:
                    x = self.ResnetLayer(inputs=x,
                                         in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=1,
                                         strides=strides,
                                         activation=activation,
                                         batch_normalization=False)
                x = add([x, y])

            in_channels = out_channels

        x = nn.BatchNorm2d()(x)
        x = F.relu(x)
        x = nn.AvgPool2d(kernel_size=4)(x)
        y = nn.Flatten()(x)
        y = nn.Linear(units=128)(y)
        y = F.relu(y)
        y = nn.Dropout(p=0.5)(y)
        y = nn.Linear(units=64)(y)
        y = F.relu(y)
        y = nn.Dropout(p=0.5)(y)
        outputs = nn.Linear(num_classes)(y)

        return F.log_softmax(outputs)
"""


