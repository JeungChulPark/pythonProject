import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#xy = []
#n_count = 1999
#x = 0.01
#a = 3
#b = 4

#for i in range(n_count):
#    new_row = [x,a*x, a*x+b]
#    xy.append(new_row)
#    x = x + 0.01

#xy = np.array(xy)
#x_data = xy[:,0]
#y_data = xy[:,2]
#plt.plot(x_data, y_data, 'bo', alpha=0.3)
#plt.show()

#X = tf.Variable(dtype="float32", name="input")
#temp_a = 0.5
#temp_b = 0.5

# a = tf.Variable(temp_a)
# b = tf.Variable(temp_b)
# y = a*X + b


X = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
Y = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

X = X[:5] + X[6:]
Y = Y[:5] + Y[6:]
"""
수식 적용
x_bar = sum(X) / len(X)
y_bar = sum(Y) / len(Y)

a = sum([(y-y_bar)*(x-x_bar) for y, x in list(zip(Y, X))])
a /= sum([(x-x_bar)**2 for x in X])
b = y_bar - a * x_bar

print('a:', a, 'b:', b)
"""
a = tf.Variable(random.random())
b = tf.Variable(random.random())

def compute_loss():
    y_pred = a * X + b
    loss = tf.reduce_mean((Y-y_pred) ** 2)
    return loss

optimizer = tf.optimizers.Adam(learning_rate=0.07)

for i in range(1000):
    optimizer.minimize(compute_loss, var_list=[a,b])

    if i % 100 == 99:
        print(i, 'a:', a.numpy(), 'b:', b.numpy(), 'loss:', compute_loss().numpy())


line_x = np.arange(min(X), max(X), 0.01)
line_y = a * line_x + b

plt.plot(line_x, line_y, 'r-')

plt.plot(X, Y, 'bo')
plt.xlabel('Population Growth Rate (%)')
plt.ylabel('Elderly Growth Rate (%)')
plt.show()