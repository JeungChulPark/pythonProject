import torch
from torch import nn

class MyLogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logistic_stack = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        prediction = self.logistic_stack(data)
        return prediction

class MyDeepLearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.deeplearning_stack = nn.Sequential(
            nn.Linear(1, 8),
            nn.Linear(8, 8),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        prediction = self.deeplearning_stack(data)
        return prediction


x_train = torch.Tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).view(10, 1)
y_train = torch.Tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).view(10, 1)

print(x_train.shape, y_train.shape)

deeplearning_model = MyDeepLearningModel()

for name, child in deeplearning_model.named_children():
    for param in child.parameters():
        print(name, param)

loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(deeplearning_model.parameters(), lr=1e-1)

nums_epoch = 5000

for epoch in range(nums_epoch+1):
    outputs = deeplearning_model(x_train)
    loss = loss_function(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch = ', epoch, ' current loss = ', loss.item())


deeplearning_model.eval()

test_data = torch.Tensor([0.5, 3.0, 3.5, 11.0, 13.0, 31.0]).view(6, 1)

pred = deeplearning_model(test_data)

logical_value = (pred > 0.5).float()

print(pred)
print(logical_value)
