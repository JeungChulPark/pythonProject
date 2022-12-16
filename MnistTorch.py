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
            mnist_img = self.ResizeImage(file)
            self.mnist_imgs.append(mnist_img)

    def __len__(self):
        return len(self.filepathes)

    def __getitem__(self, idx):
        # img = np.array(self.mnist_imgs[idx])
        # transform_img = self.transform(img)
        transform_img = self.transform(self.mnist_imgs[idx])
        return transform_img

    def ResizeImage(self, filename):
        img = cv2.imread(filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        # ret, img_th = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)
        # resized_img = cv2.resize(img_th, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        resized_img = cv2.resize(img_gray, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        # scalingFactor = 1.0/255.0
        # resized_img = np.float32(resized_img)
        # resized_img = resized_img * scalingFactor
        # resized_img = torch.Tensor(resized_img, dtype=torch.float32)
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
epochs = 10
num_classes = 10


model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

Train()
torch.save(model.state_dict(), 'save/ConvNetModel10.pt')

# model.load_state_dict(torch.load('save/ConvNetModel10.pt'))
# model.eval()
#
# transform = transforms.Compose(
#     [
#         transforms.ToTensor()
#     ]
# )
# dataset = CustomDataSet(path='Image/Result', transform=transform)
# dataloader = DataLoader(dataset, batch_size=batch_size)
#
# f = open("Image/Result/answer_ConvNet_model10.txt", 'w')
# with torch.no_grad():
#     for data in dataloader:
#         # print(type(data))
#         # print(data)
#         # data = data.to(device, dtype=torch.float32)
#         data = data.to(device)
#         out = model(data)
#         preds = torch.max(out.data, 1)[1]
#         print(preds)
#         for i in preds:
#             f.write("%d\n" % i)
# f.close()