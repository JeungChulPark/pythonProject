import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np


x_data = np.array([[1,2,0],[5,4,3],[1,2,-1],[3,1,0],[2,4,2],
                   [4,1,2],[-1,3,2],[4,3,3],[0,2,6],[2,2,1],
                   [1,-2,-2],[0,1,3],[1,1,3],[0,1,4],[2,3,3]])
t_data = np.array([-4,4,-6,3,-4,
                   9,-7,5,6,0,
                   4,3,5,5,1])

model = Sequential()
model.add(Dense(1, input_shape=(3,), activation='linear'))
model.compile(optimizer=SGD(learning_rate=1e-2), loss='mse')

model.summary()

hist = model.fit(x_data, t_data, epochs=1000)

test_data = [[5,5,0], [2,3,1],[-1,0,-1],[10,5,2],[4,-1,-2]]
ret_val = [ 2*data[0] - 3*data[1] + 2*data[2] for data in test_data ]

prediction_val = model.predict(np.array(test_data))

print(prediction_val)
print("======================")
print(ret_val)

print(model.input)
print(model.output)
print(model.weights)
