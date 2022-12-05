import tensorflow as tf
print(tf.__version__)

rand = tf.random.uniform([1], 0, 1)
print(rand)
rand = tf.random.normal([1], 0, 1)
print(rand)
