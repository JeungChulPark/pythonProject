import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / ( 1+np.exp(-x))

def numerical_derivative(f, x):
    delta_x = 1e-4

    grad = np.zeros_like(x)
    # print("debug 1. initial input variable =", x)
    # print("debug 2. initial grad =", grad)
    # print("======================================")

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        # print("debug 3. idx = ", idx, " , x[idx] = ", x[idx])
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)

        grad[idx] = (fx1-fx2)/(2*delta_x)
        # print("debug 4. grad[idx] =", grad[idx])
        # print("debug 5. grad = ", grad)
        # print("==================================")
        x[idx] = tmp_val
        it.iternext()

    return grad


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes      # input_nodes = 784
        self.hidden_nodes = hidden_nodes    # hidden_nodes = 100
        self.output_nodes = output_nodes    # output_nodes = 10

        # 2층 hidden layer unit
        self.W2 = np.random.rand(self.input_nodes, self.hidden_nodes)   # W2 = (784 X 100)
        self.b2 = np.random.rand(self.hidden_nodes)                     # b2 = (100, )

        # 3층 output layout unit
        self.W3 = np.random.rand(self.hidden_nodes, self.output_nodes)   # W3 = (100 X 10)
        self.b3 = np.random.rand(self.output_nodes)                     # b3 = (10, )

        self.__learning_rate = 1e-4

    # 손실함수
    def feed_forward(self):
        delta = 1e-7

        z1 = np.dot(self.input_nodes, self.W2) + self.b2
        y1 = sigmoid(z1)

        z2 = np.dot(y1, self.W3) + self.b3
        y = sigmoid(z2)

        # cross-entropy
        return -np.sum(self.target_data * np.log(y+delta) + (1-self.target_data)*np.log((1-y)+delta))

    # 손실 값 계산
    def loss_val(self):
        delta = 1e-7
        z1 = np.dot(self.target_data, self.W2) + self.b2
        y1 = sigmoid(z1)

        z2 = np.dot(self.target_data, self.W3) + self.b3
        y = sigmoid(z2)
        # cross-entropy
        return -np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log(1 - y + delta))

    # input_data : 784개, target_data : 10개
    def train(self, training_data):
        # normalize
        self.target_data = np.zeros(self.output_nodes) + 0.01
        self.target_data[int(training_data[0])] = 0.99

        self.input_data = (training_data[1:] / 255.0 * 0.99) + 0.01

        f = lambda x: self.feed_forward()
        self.W2 -= self.__learning_rate * numerical_derivative(f, self.W2)
        self.b2 -= self.__learning_rate * numerical_derivative(f, self.b2)
        self.W3 -= self.__learning_rate * numerical_derivative(f, self.W3)
        self.b3 -= self.__learning_rate * numerical_derivative(f, self.b3)

    # 미래 값 예측 함수
    def predict(self, input_data):
        z1 = np.dot(input_data, self.W2) + self.b2
        y1 = sigmoid(z1)
        z2 = np.dot(y1, self.W3) + self.b3
        y = sigmoid(z2)

        predicted_num = np.argmax(y)
        return predicted_num

    def accuracy(self, test_data):
        matched_list = []
        not_matched_list = []

        for index in range(len(test_data)):
            label = int(test_data[index, 0])

            # normalize
            data = (test_data[index, 1:] /255.0*0.99) + 0.1

            predicted_num = self.predict(data)

            if label == predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)
        print("Current Accuracy = ", 100*(len(matched_list)/(len(test_data))), " %")
        return matched_list, not_matched_list


training_data = np.loadtxt('./mnist_train.csv', delimiter=',', dtype=np.float32)
test_data = np.loadtxt('./mnist_test.csv', delimiter=',', dtype=np.float32)

print("training_data.shape = ", training_data.shape, " , test_data.shape = ", test_data.shape)

# img = training_data[0][1:].reshape(28,28)
#
# plt.imshow(img, cmap='gray')
# plt.show()

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)

for step in range(30001):
    index = np.random.randint(0, len(training_data)-1)

    nn.train(training_data[index])

    if step % 400 == 0:
        print("step = ", step, " , loss_val = ", nn.loss_val())
nn.accuracy(test_data)