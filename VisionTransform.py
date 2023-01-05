import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms

from model.ViT import ViT
from model.ViT import Net
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device + " is available")

learning_rate = 0.001
batch_size = 4
epochs = 100
num_classes = 10


model = ViT().to(device)
# model = Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size)
                                          # batch_size=batch_size,
                                          # shuffle=True,
                                          # num_workers=2)

for epoch in range(epochs):
    avg_cost = 0
    for data, target in trainloader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        hypothesis = model(data)
        cost = criterion(hypothesis, target)
        cost.backward()
        optimizer.step()
        avg_cost += cost / len(trainloader)
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

model.eval()

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size)
                                         # ,
                                         # shuffle=False,
                                         # num_workers=2)

with torch.no_grad():
    correct = 0
    total = 0

    for data, target in testloader:
        data.data.to(device)
        target = target.to(device)
        out = model(data)
        preds = torch.max(out.data, 1)[1]
        total += len(target)
        correct += (preds==target).sum().item()
    print("Test Accuracy: ", 100.*correct/total, '%')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# summary(ViT(), (3, 224, 224), device='cpu')