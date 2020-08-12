'''
Assignment1 - Problem 6
'''
import tensorflow as tf

with tf.compat.v1.Session() as sess:
    x = tf.constant([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    y = tf.constant([34, 28, 45, 67, 89, 93, 24, 49, 11, 7])
    writer = tf.compat.v1.summary.FileWriter('./summaries2', sess.graph)
    z = tf.add(x,y)
    res = sess.run(z)
    writer.flush()
