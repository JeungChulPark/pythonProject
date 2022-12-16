import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import __add__


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(
            reduce(
                __add__,
                [
                    (k // 2 + (k - 2 * (k // 2)) - 1, k // 2)
                    for k in self.kernel_size[::-1]
                ],
            )
        )
    def forward(self, input):
        return self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.version = 0
        if self.version == 0:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
            self.mp = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(320, 100)
            self.fc2 = nn.Linear(100, 10)
            self.drop1D = nn.Dropout(p=0.5)
            self.relu = nn.ReLU(inplace=True)

        elif self.version == 1:
            self.conv1 = Conv2dSamePadding(1, 64, kernel_size=3)
            self.conv2 = Conv2dSamePadding(64, 64, kernel_size=3)
            self.conv3 = Conv2dSamePadding(64, 128, kernel_size=3)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
            self.conv5 = Conv2dSamePadding(256, 512, kernel_size=3)
            self.conv6 = nn.Conv2d(512, 1024, kernel_size=3)
            # self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
            # self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
            # self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
            # self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
            self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
            self.mp = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(4096, 256)
            # self.fc2 = nn.Linear(1024, 512)
            # self.fc3 = nn.Linear(512, 256)
            self.fc4 = nn.Linear(256, 10)
            self.drop1D = nn.Dropout(p=0.5)
        elif self.version == 2:
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

    def forward(self, x):
        if self.version == 0:
            x = self.mp(F.relu(self.conv1(x)))
            x = self.mp(F.relu(self.conv2(x)))
            # x = self.drop2D(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            # x = self.drop1D(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
        elif self.version == 1:
            x = self.mp(F.relu(self.conv2(F.relu(self.conv1(x)))))
            # x = self.drop2D(x)
            x = self.mp(F.relu(self.conv4(F.relu(self.conv3(x)))))
            # x = self.drop2D(x)
            x = self.mp(F.relu(self.conv6(F.relu(self.conv5(x)))))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            # x = self.drop1D(x)
            # x = F.relu(self.fc2(x))
            # x = self.drop1D(x)
            # x = F.relu(self.fc3(x))
            # x = self.drop1D(x)
            x = F.relu(self.fc4(x))
            return F.log_softmax(x, dim=1)
        elif self.version == 2:
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
