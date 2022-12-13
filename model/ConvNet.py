import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.version = 1
        if self.version == 0:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
            self.mp = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(320, 100)
            self.fc2 = nn.Linear(100, 10)
        elif self.version == 1:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=[3//2, 3//2], padding_mode='replicate')
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=[1,1], padding_mode='replicate')
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=[1,1], padding_mode='replicate')
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
            self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=[1,1], padding_mode='replicate')
            self.conv6 = nn.Conv2d(512, 1024, kernel_size=3)
            self.mp = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(4096, 512)
            # self.fc2 = nn.Linear(2048, 1024)
            # self.fc3 = nn.Linear(1024, 512)
            self.fc4 = nn.Linear(512, 10)
            self.drop2D = nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, x):
        if self.version == 0:
            x = F.relu(self.mp(self.conv1(x)))
            x = F.relu(self.mp(self.conv2(x)))
            x = self.drop2D(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.fc2(x)
            return F.log_softmax(x)
        elif self.version == 1:
            x = self.mp(F.relu(self.conv2(F.relu(self.conv1(x)))))
            x = self.mp(F.relu(self.conv4(F.relu(self.conv3(x)))))
            x = self.mp(F.relu(self.conv6(F.relu(self.conv5(x)))))
            x = self.drop2D(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            # x = self.fc2(x)
            # x = self.fc3(x)
            x = self.fc4(x)
            return F.log_softmax(x)
