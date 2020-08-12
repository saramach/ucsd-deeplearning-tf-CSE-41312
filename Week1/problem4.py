'''
Assignment1 - Problem 4
'''
import tensorflow as tf
x1 = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(x1)
reshapedx1 = tf.reshape(x1, [6,2], 'reshaped-x1')
print(reshapedx1)

