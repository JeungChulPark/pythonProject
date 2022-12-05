import numpy as np

loaded_data = np.loadtxt('./data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[:, 0:-1]
t_data = loaded_data[:, [-1]]

W = np.random.rand(3,1)
b = np.random.rand(1)
print("W = ", W, ": , W.shape = ", W.shape, " , b = ", b, " , b.shape = ", b.shape)

# W = [[0.04946736][0.00916638][0.56439521]], W.shape=(3,1), b=[0.34662569], b.shape=(1,)

def loss_func(x,t):
    y = np.dot(x, W) + b
    return ( np.sum((t-y)**2)) / (len(x))


def error_val(x, t):
    y = np.dot(x,W) + b
    return ( np.sum((t-y)**2)) / (len(x))


def predict(x):
    y = np.dot(x, W) + b
    return y

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

learning_rate = 1e-5

f = lambda x : loss_func(x_data, t_data)
print("Initial error value = ", error_val(x_data, t_data), "Initial W = ", "\n", " , b = ", b)

for step in range(10001):
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)

    if(step % 400 == 0):
        print("step = ", step, "error value = ", error_val(x_data, t_data), "W = ", W, " , b = ", b)

print("Predict : ")
test_data = np.array([100, 98, 81])
print(test_data)
print(predict(test_data))