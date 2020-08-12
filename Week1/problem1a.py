'''
Assignment 1 - Problem #1
Executing tensor addition in lazy(default) mode
'''
import tensorflow as tf

with tf.compat.v1.Session() as sess:
    x = tf.constant([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    y = tf.constant([34, 28, 45, 67, 89, 93, 24, 49, 11, 7])

    z = tf.add(x,y)

    result = sess.run(z)
    print(result)
	

