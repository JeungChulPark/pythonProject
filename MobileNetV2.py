import torch
import os
import cv2
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.mobilenet import mobilenet_v2
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
import pandas as pd

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
        # path = "Image/result"
        filelist = []

        for root, dirs, files in os.walk(path):
            for file in files:
                filelist.append(os.path.join(root, file))
        
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

def train(model, device, train_loader, optimizer, epoch):
    log_interval = 10
    loss_func = CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.repeat(1, 3, 1, 1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()
            ))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.repeat(1, 3, 1, 1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


def main():
    batch_size = 1000
    learning_rate = 1.0
    reduce_lr_gamma = 0.7
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device : {} Epoches: {} Batch size : {}'.format(device, epochs, batch_size))

    kwargs = {'batch_size': batch_size}
    if torch.cuda.is_available():
        kwargs.update({'num_workers': 1, 'pin_memory': True})

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))
    ])

    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    print('Length train: {} Length test: {}'.format(len(dataset1), len(dataset2)))

    train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, shuffle=False, **kwargs)
    print('Number of train batches: {} Number of test batches: {}'.format(len(train_loader), len(test_loader)))

    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
    model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=reduce_lr_gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "./save/mnist_modelnet_v2.pt")

    ids = list(range(len(dataset2)))
    submission = pd.DataFrame(ids, columns=['id'])
    predictions = []
    real = []
    for data, target in test_loader:
        data = data.repeat(1, 3, 1, 1)
        data = data.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        predictions += list(pred.cpu().numpy()[:, 0])
        real += list(target.numpy())
    submission['pred'] = predictions
    submission['real'] = real
    submission.to_csv('submission.csv', index=False)
    print('Submission saved in: {}'.format('submission.csv'))

def custummain():
    learning_rate = 1.0
    reduce_lr_gamma = 0.7
    batch_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mobilenet_v2().to(device)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=reduce_lr_gamma)

    model.load_state_dict(torch.load('./save/mnist_modelnet_v2.pt'))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))
    ])

    dataset = CustomDataSet(path='Image/result', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    res = []
    f = open("Image/result/answer_mobilenet_v2_model100.txt", 'w')
    with torch.no_grad():
        for data in dataloader:
            data = data.repeat(1, 3, 1, 1)
            data = data.to(device)
            out = model(data)
            preds = torch.max(out.data, 1)[1]
            res.append(preds.to('cpu'))
            for i in preds:
                f.write("%d\n" % i)
    f.close()


if __name__ == '__main__':
#   main()
    custummain()

