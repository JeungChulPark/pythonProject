import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from model.ConvNet import ConvNet
from torch.utils.data import Dataset, DataLoader

class CustomDataSet(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.filepathes = self.GetFilePath(path)
        self.mnist_imgs = []
        for file in self.filepathes:
            # print(file)
            mnist_img = self.ResizeImage(file)
            self.mnist_imgs.append(mnist_img)
        # self.transform_imgs = np.array(self.mnist_imgs)
        #self.transform_imgs = self.transform_imgs / 255.0
        #self.transform_imgs = self.transform(self.transform_imgs)

    def __len__(self):
        print("__len__")
        return len(self.filepathes)

    def __getitem__(self, idx):
        # print("get item : %d" % idx)
        # if self.transform is not None:
        # self.transform_imgs[idx] = self.transform_imgs[idx] / 255.0
        # self.transform_imgs[idx] = self.transform(self.transform_imgs[idx])
        # return self.transform_imgs[idx]
        # return self.mnist_imgs[idx]

        img = np.array(self.mnist_imgs[idx])
        # img = img.reshape(1, 28, 28)
        img = img / 255.0
        transform_img = self.transform(img)
        return transform_img
        # return self.mnist_imgs[idx]

    def ResizeImage(self, filename):
        img = cv2.imread(filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        ret, img_th = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)
        resized_img = cv2.resize(img_th, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        return resized_img

    def GetFilePath(self, path):
        # path = "Image/Result"
        filelist = []

        for root, dirs, files in os.walk(path):
            for file in files:
                filelist.append(os.path.join(root, file))

        print("\n")
        v = [x for x in filelist if x.endswith(".png")]
        return v

def GetFilePath():
    path = "Image/Result"
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    # print("\n")
    v = [x for x in filelist if x.endswith(".png")]
    return v

def ResizeImage(filename):
    img = cv2.imread(filename)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    ret, img_th = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)

    resized_img = cv2.resize(img_th, dsize=(28, 28), interpolation=cv2.INTER_AREA)
    return resized_img

def DataTrainLoad():
    train_set = torchvision.datasets.MNIST(
        root = './data/MNIST',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_loader = DataLoader(train_set, batch_size=batch_size)
    return train_loader
def DataTestLoad():
    test_set = torchvision.datasets.MNIST(
        root='./data/MNIST',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    test_loader = DataLoader(test_set, batch_size=batch_size)
    return test_loader

def Train():
    train_loader = DataTrainLoad()
    for epoch in range(epochs):
        avg_cost = 0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            hypothesis = model(data)
            cost = criterion(hypothesis, target)
            cost.backward()
            optimizer.step()
            avg_cost += cost / len(train_loader)
        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

    model.eval()

    test_loader = DataTestLoad()
    with torch.no_grad():
        correct = 0
        total = 0

        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            preds = torch.max(out.data, 1)[1]
            total += len(target)
            correct += (preds==target).sum().item()
        print("Test Accuracy: ", 100.*correct/total, '%')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device + " is available")

learning_rate = 0.001
batch_size = 100
num_classes = 10
epochs = 100

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train()
# torch.save(model.state_dict(), 'save/model100.pt')

model.load_state_dict(torch.load('save/model100.pt'))
model.eval()

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
dataset = CustomDataSet(path='Image/Result', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size)
print('dataloader : ')

with torch.no_grad():
    for data in dataloader:
        # print(type(data))
        # print(data)
        data = data.to(device, dtype=torch.float32)
        out = model(data)
        preds = torch.max(out.data, 1)[1]
        print(preds)
        # for img in data:
        #     cv2.imshow("img", img)
        #     cv2.waitKey(0)
        #     break


# test_loader = DataTestLoad()
# with torch.no_grad():
#     correct = 0
#     total = 0
#
#     for data, target in test_loader:
#         print(type(data))
#         data = data.to(device)
#         target = target.to(device)
#         out = model(data)
#         preds = torch.max(out.data, 1)[1]
#         print(preds)
#         total += len(target)
#         correct += (preds == target).sum().item()
#     print("Test Accuracy: ", 100.*correct/total, '%')

