'''
Assignment 1 - Problem #1(b)
Executing tensor addition in eager mode
'''
import tensorflow as tf

# In Tensorflow 2, the eager execution is enabled by default

x = tf.constant([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
y = tf.constant([34, 28, 45, 67, 89, 93, 24, 49, 11, 7])
z = tf.add(x,y)
print(z)

