import time

import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.datasets
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from torchvision.datasets import MNIST
from tqdm import tqdm_notebook
from tqdm import tqdm
from time import sleep
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r+h_r)
        inputgate = F.sigmoid(i_i+h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)
        return hy


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []
        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self.fc(out)
        return out

class GRU(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        output, (hn) = self.gru(x, (h_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

writer = SummaryWriter()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(125)

if torch.cuda.is_available() :
    torch.cuda.manual_seed_all(125)

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (1.0, ))
])

download_root = './data/MNIST'

train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)

batch_size = 64

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

batch_size = 100
n_iters = 6000
num_epochs = n_iters / (len(train_dataset)/batch_size)
num_epochs = int(num_epochs)

input_dim = 28
hidden_dim = 128
layer_dim = 1
output_dim = 10

model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train(model, train_loader):
    seq_dim = 28
    loss_list = []
    # iter = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            avg_train_loss = 0
            current_lr = get_lr(optimizer)
            print('[Epoch: {}/{}, current lr = {}]'.format(epoch+1, num_epochs, current_lr))
            for images, labels in tepoch:
                if torch.cuda.is_available():
                    images = Variable(images.view(-1, seq_dim, input_dim).cuda())
                    labels = Variable(labels.cuda())
                else:
                    images = Variable(images.view(-1, seq_dim, input_dim))
                    labels = Variable(labels)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                writer.add_scalar("GRU/loss/train", loss, epoch)
                if torch.cuda.is_available():
                    loss.cuda()
                loss.backward()
                optimizer.step()

                avg_train_loss += loss.item()
                loss_list.append(loss.item())

            avg_train_loss /= len(train_loader)
            # iter += 1
            # if iter % 500 == 0:
            sleep(0.1)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            with tqdm(valid_loader, unit="batch") as tepoch:
                avg_val_loss = 0
                for images, labels in tepoch:
                    if torch.cuda.is_available():
                        images = Variable(images.view(-1, seq_dim, input_dim).cuda())
                    else:
                        images = Variable(images.view(-1, seq_dim, input_dim))

                    target = labels.to(device)
                    outputs = model(images)
                    val_loss = criterion(outputs, target)
                    writer.add_scalar("GRU/loss/val", val_loss, epoch)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)

                    if torch.cuda.is_available():
                        correct += (predicted.cpu() == labels.cpu()).sum()
                    else:
                        correct += (predicted == labels).sum()
                    writer.add_scalar("GRU/accuracy/val", 100*correct, epoch)
                    avg_val_loss += val_loss.item()

                avg_val_loss /= len(valid_loader)
                accuracy = 100 * correct / total
                print("train loss : %.6f, val loss : %.6f, Accuracy : %.2f, time : %.4f min" %
                      (avg_train_loss, avg_val_loss, accuracy, (time.time() - start_time)/60))
                print('-' * 10)
                sleep(0.1)

    return loss.item(), accuracy

def evaluate(model, val_iter):
    seq_dim = 28
    corrects, total, total_loss = 0, 0, 0
    model.eval()
    for images, labels in val_iter:
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_dim, input_dim).cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim))

        logit = model(images).to(device)
        labels = labels.to(device)
        loss = F.cross_entropy(logit, labels, reduction='sum')
        _, predicted = torch.max(logit.data, 1)
        total += labels.size(0)
        total_loss += loss.item()
        corrects += (predicted == labels).sum()

    avg_loss = total_loss / len(val_iter.dataset)
    avg_accuracy = corrects / total
    return avg_loss, avg_accuracy

train_loss, train_acc = train(model, train_loader)
test_loss, test_acc = evaluate(model, test_loader)
print(f'Test Loss : {test_loss:5.2f} | Test Accuracy : {test_acc:5.2f}')

writer.close()