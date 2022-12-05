import numpy
import numpy as np

x_data = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(10,1)
t_data = np.array([0,0,0,0,0,0,1,1,1,1]).reshape(10,1)

W = np.random.rand(1,1)
b = np.random.rand(1)
print("W = ", W, " , W.shape = ", W.shape, " , b = ", b, " , b.shape = ", b.shape)

def sigmoid(x):
    return 1 / ( 1+numpy.exp(-x))

def loss_func(x, t):
    delta = 1e-7
    z=numpy.dot(x, W) + b
    y = sigmoid(z)
    return -numpy.sum( t*numpy.log(y+delta) + (1-t)*numpy.log(1-y+delta))

def error_val(x, t):
    delta = 1e-7
    z = numpy.dot(x, W) + b
    y = sigmoid(z)
    # cross-entropy
    return -numpy.sum(t * numpy.log(y + delta) + (1 - t) * numpy.log(1 - y + delta))

def predict(x):
    z = np.dot(x, W) + b
    y = sigmoid(z)

    if y > 0.5:
        result = 1
    else:
        result = 0
    return y, result

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

learning_rate = 1e-2

f = lambda x : loss_func(x_data, t_data)
print("Initial error value = ", error_val(x_data, t_data), "Initial W = ", "\n", " , b = ", b)

for step in range(10001):
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)

    if(step % 400 == 0):
        print("step = ", step, "error value = ", error_val(x_data, t_data), "W = ", W, " , b = ", b)

(real_val, logical_val) = predict(12.9)
print(real_val, logical_val)