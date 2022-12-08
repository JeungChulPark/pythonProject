import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model.ConvNet import ConvNet

def GetFilePath():
    path = "Image/Result"
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    print("\n")
    v = [x for x in filelist if x.endswith(".png")]
    return v

def ResizeImage(filename):
    img = cv2.imread(filename)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    ret, img_th = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)

    resized_img = cv2.resize(img_th, dsize=(28, 28), interpolation=cv2.INTER_AREA)
    return resized_img


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device + " is available")

learning_rate = 0.001
batch_size = 100
num_classes = 10
epochs = 2

def DataTrainLoad():
    train_set = torchvision.datasets.MNIST(
        root = './data/MNIST',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
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

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    return test_loader
# examples = enumerate(train_set)
# batch_idx, (example_data, example_targets) = next(examples)
# example_data.shape


model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



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

# torch.save(model.state_dict(), 'save/model.pt')


model = ConvNet()
model.load_state_dict(torch.load('save/model.pt'))
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
        correct += (preds == target).sum().item()
    print("Test Accuracy: ", 100.*correct/total, '%')


# v = GetFilePath()
# img_array = []
# for file in v:
#     strings = file.split('\\')
#     if strings[1] == "subject_4":
#         break
#     mnist_imgs = ResizeImage(file)
#     img_array.append(mnist_imgs)


