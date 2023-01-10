import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import copy
import time
import numpy as np
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
# %matplotlib inline
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.ConvNet import ConvNet
from model.ResNetTorch import ResNetTorch
from model.ResnetBlockLayer import ResnetBlockLayer
from tqdm import tqdm
from time import sleep


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

def Train(model, params):

    for epoch in range(epochs):
        with tqdm(train_dl, unit="batch") as tepoch:
            avg_cost = 0
            for data, target in tepoch:
                data = data.to(device)
                target = target.to(device)
                opt.zero_grad()
                hypothesis = model(data)
                cost = loss_func(hypothesis, target)
                cost.backward()
                opt.step()
                avg_cost += cost / len(train_dl)
            print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
            sleep(0.1)

    model.eval()

    val_dl = DataTestLoad()
    with torch.no_grad():
        correct = 0
        total = 0

        for data, target in val_dl:
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            preds = torch.max(out.data, 1)[1]
            total += len(target)
            correct += (preds==target).sum().item()
        print("Test Accuracy: ", 100.*correct/total, '%')


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    with tqdm(dataset_dl, unit="batch") as tepoch:
        for xb, yb in tepoch:
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)

            loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

            running_loss += loss_b

            if metric_b is not None:
                running_metric += metric_b

            if sanity_check is True:
                break
        sleep(0.1)
    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric

def train_val(model, params):
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_dl = params['train_dl']
    val_dl = params['val_dl']
    sanity_check = params['sanity_check']
    lr_scheduler = params['lr_scheduler']
    path2weights = params['path2weights']

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr = {}'.format(epoch+1, num_epochs, current_lr))
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print('Copied best model weights')

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print('Loading data model weights!')
            model.load_state_dict(best_model_wts)
        print('train loss : %.6f, val loss : %.6f, accuracy : %.2f, time : %.4f min' %
              (train_loss, val_loss, 100*val_metric, (time.time() - start_time)/60))
        print('-'*10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device + " is available")

learning_rate = 0.001
batch_size = 100
epochs = 10
num_classes = 10

n = 3
version = 3
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2
elif version == 3:
    depth = n * 6 + 2
elif version == 4:
    depth = n * 9 + 2

teacher = ResNetTorch(version=version, layer=ResnetBlockLayer, layeriter=None, depth=depth, num_classes=num_classes).to(device)
loss_func = nn.CrossEntropyLoss().to(device)
opt = optim.Adam(teacher.parameters(), lr = learning_rate)

lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

# def initialize_weights(model):
#     classname = model.__class__.__name__
#
#     if classname.find('Linear') != -1:
#         nn.init.normal_(model.weight.data, 0.0, 0.02)
#         nn.init.constant_(model.bias.data, 0)
#
#     elif classname.find('BatchNorm') != -1:
#             nn.init.normal_(model.weight.data, 0.0, 0.02)
#             nn.init.constant_(model.bias.data, 0)
#
# teacher.apply(initialize_weights)

train_dl = DataTrainLoad()
val_dl = DataTestLoad()

params_train = {
    'num_epochs' : 100,
    'optimizer':opt,
    'loss_func': loss_func,
    'train_dl': train_dl,
    'val_dl': val_dl,
    'sanity_check': False,
    'lr_scheduler': lr_scheduler,
    'path2weights':'./save/teacher_weights.pt'
}

# teacher, loss_hist, metric_hist = train_val(teacher, params_train)
# num_epochs = params_train['num_epochs']
# draw_train_val(num_epochs, loss_hist, metric_hist)

teacher.load_state_dict(torch.load('./save/ResNet_V3_Model100.pt'))
student = ConvNet(0).to(device)

opt = optim.Adam(student.parameters(), lr = learning_rate)

def distillation(y, labels, teacher_scores, T, alpha):
    '''
    distillation loss + classification loss
    y: student
    labels: hard label
    teacher_scores: soft label
    '''
    student_loss = F.cross_entropy(input=y, target=labels)
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(
                        F.log_softmax(y/T, dim=1),
                        # F.softmax(teacher_scores/T, dim=1) * (T*T*2.0+alpha) +
                        F.softmax(teacher_scores / T, dim=1)) * (T * T * alpha)
    total_loss = alpha*student_loss + (1.0 - alpha) * distillation_loss
    return total_loss

def distill_lossbatch(output, target, teacher_output, loss_fn=distillation, opt=opt):
    loss_b = loss_fn(output, target, teacher_output, T=10.0, alpha=0.1)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()
    return loss_b.item(), metric_b

params_distill_train = {
    'num_epochs' : 100,
    'optimizer':opt,
    'loss_func': loss_func,
    'train_dl': train_dl,
    'val_dl': val_dl,
    'sanity_check': False,
    'lr_scheduler': lr_scheduler,
    'path2weights':'./save/student_weights.pt'
}

def train_distill_val(teacher, student, params):
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_dl = params['train_dl']
    val_dl = params['val_dl']
    sanity_check = params['sanity_check']
    lr_scheduler = params['lr_scheduler']
    path2weights = params['path2weights']

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(student.state_dict())
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr = {}'.format(epoch+1, num_epochs, current_lr))
        student.train()

        running_loss = 0.0
        running_metric = 0.0
        len_data = len(train_dl.dataset)

        with tqdm(train_dl, unit="batch") as tepoch:
            for xb, yb in tepoch:
                xb = xb.to(device)
                yb = yb.to(device)

                output = student(xb)
                teacher_output = teacher(xb).detach()
                loss_b, metric_b = distill_lossbatch(output, yb, teacher_output, loss_fn=distillation, opt=opt)

                running_loss += loss_b
                running_metric += metric_b
            sleep(0.1)

        train_loss = running_loss / len_data
        train_metric = running_metric / len_data

        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        student.eval()

        with torch.no_grad():
            val_loss, val_metric = loss_epoch(student, loss_func, val_dl)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(student.state_dict())
            torch.save(student.state_dict(), path2weights)
            print('Copied best model weights')

        lr_scheduler.step(val_loss)

        if current_lr != get_lr(opt):
            print('Loading data model weights!')
            student.load_state_dict(best_model_wts)

        print('train loss : %.6f, val loss : %.6f, accuracy : %.2f, time : %.4f min' %
              (train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        print('-'*10)

    return teacher, student, loss_history, metric_history

teacher, student, loss_hist, metric_hist = train_distill_val(teacher, student, params_distill_train)
num_epochs = params_distill_train['num_epochs']
draw_train_val(num_epochs, loss_hist, metric_hist)