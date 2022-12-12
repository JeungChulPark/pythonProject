import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        version = 1
        if version == 0:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

            self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
            self.mp = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(320, 100)
            self.fc2 = nn.Linear(100, 10)
        elif version == 1:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding='same')
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding_mode='same')
            self.mp1 = nn.MaxPool2d(2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding_mode='same')
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding_mode='valid')
            self.mp2 = nn.MaxPool2d(2)
            self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding_mode='same')
            self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding_mode='valid')
            self.mp2 = nn.MaxPool2d(2)
            self.drop2D = nn.Dropout2d(p=0.25, inplace=False)

            self.fc1 = nn.Linear(320, 100)
            self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = self.drop2D(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x)