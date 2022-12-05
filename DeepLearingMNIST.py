import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / ( 1+np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes      # input_nodes = 784
        self.hidden_nodes = hidden_nodes    # hidden_nodes = 100
        self.output_nodes = output_nodes    # output_nodes = 10

        # 2층 hidden layer unit
        self.W2 = np.random.rand(self.input_nodes, self.hidden_nodes) / np.sqrt(self.input_nodes/2)  # W2 = (784 X 100)
        self.b2 = np.random.rand(self.hidden_nodes)                     # b2 = (100, )

        # 3층 output layout unit
        self.W3 = np.random.rand(self.hidden_nodes, self.output_nodes) / np.sqrt(self.input_nodes/2)  # W3 = (100 X 10)
        self.b3 = np.random.rand(self.output_nodes)                     # b3 = (10, )

        self.Z3 = np.zeros([1, output_nodes])
        self.A3 = np.zeros([1, output_nodes])

        self.Z2 = np.zeros([1, hidden_nodes])
        self.A2 = np.zeros([1, hidden_nodes])

        self.Z1 = np.zeros([1, input_nodes])
        self.A1 = np.zeros([1, input_nodes])
        self.learning_rate = learning_rate

    # 손실함수
    def feed_forward(self):
        delta = 1e-7

        self.Z1 = self.input_data
        self.A1 = self.input_data

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = sigmoid(self.Z3)

        # cross-entropy
        return -np.sum(self.target_data * np.log(self.A3+delta) + (1-self.target_data)*np.log((1-self.A3)+delta))

    # 손실 값 계산
    def loss_val(self):
        delta = 1e-7
        self.Z1 = self.input_data
        self.A1 = self.input_data

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = sigmoid(self.Z3)
        # cross-entropy
        return -np.sum(self.target_data * np.log(self.A3 + delta) + (1 - self.target_data) * np.log(1 - self.A3 + delta))

    # input_data : 784개, target_data : 10개
    def train(self, input_data, target_data):
        # normalize
        self.target_data = target_data
        self.input_data = input_data

        loss_val = self.feed_forward()

        loss_3 = (self.A3-self.target_data) * self.A3 * (1-self.A3)

        self.W3 = self.W3 - self.learning_rate * np.dot(self.A2.T, loss_3)
        self.b3 = self.b3 - self.learning_rate * loss_3

        loss_2 = np.dot(loss_3, self.W3.T) * self.A2 * (1-self.A2)

        self.W2 = self.W2 - self.learning_rate * np.dot(self.A1.T, loss_2)
        self.b2 = self.b2 - self.learning_rate * loss_2

    # 미래 값 예측 함수
    def predict(self, input_data):
        Z2 = np.dot(input_data, self.W2) + self.b2
        A2 = sigmoid(Z2)
        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = sigmoid(Z3)

        predicted_num = np.argmax(A3)
        return predicted_num

    def accuracy(self, test_data):
        matched_list = []
        not_matched_list = []

        for index in range(len(test_data)):
            label = int(test_data[index, 0])

            # normalize
            data = (test_data[index, 1:] /255.0*0.99) + 0.1

            predicted_num = self.predict(np.array(data, ndmin=2))

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
learning_rate = 0.3
epochs = 1

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#start_time = datetime.now()
for i in range(epochs):
    for step in range(len(training_data)):
        target_data = np.zeros(output_nodes) + 0.01
        target_data[int(training_data[step, 0])] = 0.99

        input_data = ((training_data[step, 1:]/255.0)*0.99)+0.01

        nn.train(np.array(input_data, ndmin=2), np.array(target_data, ndmin=2))

        if step % 400 == 0:
            print("step = ", step, " , loss_val = ", nn.loss_val())

nn.accuracy(test_data)