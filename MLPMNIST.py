import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

class MyDeepLearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, data):
        data = self.flatten(data)
        data = self.fc1(data)
        data = self.relu(data)
        data = self.dropout(data)
        logits = self.fc2(data)
        return logits

def model_train(dataloader, model, loss_function, optimizer):
    model.train()

    train_loss_sum = train_correct = train_total = 0

    total_train_batch = len(dataloader)

    for images, labels in dataloader:
        x_train = images.view(-1, 28*28)
        y_train = labels

        outputs = model(x_train)
        loss = loss_function(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

        train_total += y_train.size(0)
        train_correct += ((torch.argmax(outputs, 1)==y_train)).sum().item()

    train_avg_loss = train_loss_sum / total_train_batch
    train_avg_accuracy = 100 * train_correct / train_total

    return (train_avg_loss, train_avg_accuracy)


def model_evaluate(dataloader, model, loss_function, optimizer):
    model.eval()

    with torch.no_grad():

        val_loss_sum = val_correct = val_total = 0

        total_val_batch = len(dataloader)

        for images, labels in dataloader:
            x_val = images.view(-1, 28 * 28)
            y_val = labels

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            val_loss_sum += loss.item()

            val_total += y_val.size(0)
            val_correct += ((torch.argmax(outputs, 1) == y_val)).sum().item()

    val_avg_loss = val_loss_sum / total_val_batch
    val_avg_accuracy = 100 * val_correct / val_total

    return (val_avg_loss, val_avg_accuracy)



def model_test(dataloader, model):
    model.eval()

    with torch.no_grad():

        test_loss_sum = test_correct = test_total = 0

        total_test_batch = len(dataloader)

        for images, labels in dataloader:
            x_test = images.view(-1, 28 * 28)
            y_test = labels

            outputs = model(x_test)
            loss = loss_function(outputs, y_test)

            test_loss_sum += loss.item()

            test_total += y_test.size(0)
            test_correct += ((torch.argmax(outputs, 1) == y_test)).sum().item()

        test_avg_loss = test_loss_sum / total_test_batch
        test_avg_accuracy = 100 * test_correct / test_total

        print('loss : ', test_avg_loss)
        print('accuracy : ', test_avg_accuracy)


train_dataset = torchvision.datasets.MNIST(
    root='./data/MNIST/',
    train=True,
    transform=transforms.ToTensor(),
    download=True)

test_dataset = torchvision.datasets.MNIST(
    root='./data/MNIST/',
    train=False,
    transform=transforms.ToTensor(),
    download=True)

print(len(train_dataset))

train_dataset_size = int(len(train_dataset) * 0.85)
validation_dataset_size = int(len(train_dataset) * 0.15)

train_dataset, validation_dataset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])

print(len(train_dataset), len(validation_dataset), len(test_dataset))


BATCH_SIZE = 32
train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataset_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MyDeepLearningModel()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []

EPOCHS = 20
for epoch in range(EPOCHS):
    train_avg_loss, train_avg_accuracy = model_train(train_dataset_loader, model, loss_function, optimizer)
    train_loss_list.append(train_avg_loss)
    train_accuracy_list.append(train_avg_accuracy)

    val_avg_loss, val_avg_accuracy = model_evaluate(validation_dataset_loader, model, loss_function, optimizer)
    val_loss_list.append(val_avg_loss)
    val_accuracy_list.append(val_avg_accuracy)

    print("epoch : %d, train loss = %.6f, train accuracy = %.6f, validation loss = %.6f, validation accuracy = %.6f "
          % (epoch, train_avg_loss, train_avg_accuracy, val_avg_loss, val_avg_accuracy))

model_test(test_dataset_loader, model)