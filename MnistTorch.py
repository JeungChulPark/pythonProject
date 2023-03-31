import copy
import os
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from model.ConvNet import ConvNet
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.ResNetTorch import ResNetTorch
from model.ResnetLayer import ResnetLayer, ResnetLayerIter
from model.ResnetLayerV2 import ResnetLayerV2, ResnetLayerV2Iter
from model.ResnetBlockLayer import ResnetBlockLayer
from model.ViT import ViT
from tqdm import tqdm
from time import sleep
from torch.utils.tensorboard import SummaryWriter
class CustomDataSet(Dataset):
    def __init__(self, target=None, path='', transform=None):
        self.path = path
        self.transform = transform
        self.filepathes = self.GetFilePath(path)
        self.mnist_imgs = []
        for file in self.filepathes:
            mnist_img = self.ResizeImage(file)
            self.mnist_imgs.append(mnist_img)
        self.target = target
        if target is not None:
            i = 0
            self.img_array_tl = []
            self.img_array_tl_res = []
            for file in self.filepathes:
                strings = file.split('\\')
                if strings[1] == "subject_4":
                    break
                if torch.tensor(np.array(int(strings[-2]))) == target[i]:
                    self.img_array_tl.append(self.mnist_imgs[i])
                    self.img_array_tl_res.append(target[i])
                i = i + 1
    def __len__(self):
        if self.target is not None:
            return len(self.img_array_tl)
        else:
            return len(self.mnist_imgs)

    def __getitem__(self, idx):
        if self.target is not None:
            transform_img = self.transform(self.img_array_tl[idx])
            label = self.img_array_tl_res[idx]
            return transform_img, label
        else:
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
        resized_img = cv2.resize(img_blur, dsize=(28, 28), interpolation=cv2.INTER_AREA)
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

def lr_schedule(self, epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate : ', lr)
    return lr

def draw_train_val(num_epochs, loss_hist, metric_hist):
    plt.title('Train-Val Loss')
    plt.plot(range(1, num_epochs + 1), loss_hist['train'], label='train')
    plt.plot(range(1, num_epochs + 1), loss_hist['val'], label='val')
    plt.ylabel('Loss')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.show()

    # plot train-val accuracy
    plt.title('Train-Val Accuracy')
    plt.plot(range(1, num_epochs + 1), metric_hist['train'], label='train')
    plt.plot(range(1, num_epochs + 1), metric_hist['val'], label='val')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.show()

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
def Train(dataloader=None, path2weights=''):
    train_loader = DataTrainLoad()
    test_loader = DataTestLoad()

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            avg_train_loss = 0
            avg_train_corrects = 0
            current_lr = get_lr(optimizer)
            print('[Epoch: {}/{}, current lr = {}'.format(epoch + 1, epochs, current_lr))
            for data, target in tepoch:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                hypothesis = model(data)
                train_loss = criterion(hypothesis, target)
                pred = hypothesis.argmax(1, keepdim=True)
                corrects = pred.eq(target.view_as(pred)).sum().item()
                train_loss.backward()
                optimizer.step()
                avg_train_loss += train_loss.item()
                avg_train_corrects += corrects

            avg_train_loss /= len(train_loader)
            avg_train_corrects /=  len(train_loader)
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Accuracy/train", avg_train_corrects, epoch)
            loss_history['train'].append(avg_train_loss)
            metric_history['train'].append(avg_train_corrects)

            if dataloader is not None:
                for data, target in dataloader:
                    data = data.to(device)
                    target = target.to(device)
                    optimizer.zero_grad()
                    hypothesis = model(data)
                    train_loss = criterion(hypothesis, target)
                    train_loss.backward()
                    optimizer.step()
                    avg_train_loss += train_loss.item()
            sleep(0.1)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            with tqdm(test_loader, unit="batch") as tepoch:
                avg_val_loss = 0
                for data, target in tepoch:
                    data = data.to(device)
                    target = target.to(device)
                    out = model(data)
                    val_loss = criterion(out, target)

                    preds = torch.max(out.data, 1)[1]
                    total += len(target)
                    correct += (preds==target).sum().item()
                    avg_val_loss += val_loss.item()

                avg_val_loss /= len(test_loader)
                avg_val_corrects = 100 * correct / total
                writer.add_scalar("Loss/val", avg_val_loss, epoch)
                writer.add_scalar("Accuracy/val", avg_val_corrects, epoch)
                loss_history['val'].append(avg_val_loss)
                metric_history['val'].append(avg_val_corrects)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), path2weights)
                    print('Copied best model weights')

                lr_scheduler.step(val_loss)
                if current_lr != get_lr(optimizer):
                    print('Loading data model weights!')
                    model.load_state_dict(best_model_wts)

                print("train loss : %.6f, val loss : %.6f, accuracy : %.2f, time : %.4f min" %
                      (avg_train_loss, avg_val_loss, avg_val_corrects, (time.time() - start_time)/60))
                print('-' * 10)
                sleep(0.1)

    model.load_state_dict(best_model_wts)
    return loss_history, metric_history

writer = SummaryWriter()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device + " is available")

learning_rate = 0.001
batch_size = 100
epochs = 100
num_classes = 10

model = ConvNet(1).to(device)
# model = ViT().to(device)
# n = 3
# version = 3
# if version == 1:
#     depth = n * 6 + 2
# elif version == 2:
#     depth = n * 9 + 2
# elif version == 3:
#     depth = n * 6 + 2
# elif version == 4:
#     depth = n * 9 + 2
#
# model = ResNetTorch(version=version, layer=ResnetBlockLayer, layeriter=None, depth=depth, num_classes=num_classes).to(device)
# model = ResNetTorch(version=version, layer=ResnetLayer, layeriter=ResnetLayerIter, depth=depth, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# loss_hist, metric_hist = Train(dataloader=None, path2weights='./save/ConvNet_V1_Model100.pt')
# writer.flush()
# draw_train_val(epochs, loss_hist, metric_hist)

model.load_state_dict(torch.load('save/Transformer_ConvNet_V1_Distillation_Model100.pt'))
model.eval()

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
dataset = CustomDataSet(path='Image/Result', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size)

res = []
f = open("Image/Result/answer_Transformer_ConvNet_V1_Distillation_Model100.txt", 'w')
with torch.no_grad():
    for data in dataloader:
        data = data.to(device)
        out = model(data)
        preds = torch.max(out.data, 1)[1]
        res.append(preds.to('cpu'))
        # res = np.append(res, preds.to('cpu'))
        # print(preds)
        for i in preds:
            f.write("%d\n" % i)
f.close()


# for i in range(len(res)):
#     if i == 0:
#         tmp = res[i]
#     else:
#         tmp2 = res[i]
#         tmp = torch.cat((tmp, tmp2), dim=0)
# # print(tmp)
# dataset = CustomDataSet(path='Image/Result', target=tmp, transform=transform)
# dataloader = DataLoader(dataset, batch_size=batch_size)
#
# Train(dataloader)
# torch.save(model.state_dict(), 'save/ConvNet_V2_Model100_tl.pt')
writer.close()