'''
Assignment1 - Problem 7
'''
import tensorflow as tf

A = tf.constant([[4, -2, 1],[6, 8, -5],[7, 9, 10]])
B = tf.constant([[6, 9, -4],[7, 5, 3],[-8, 2, 1]])
C = tf.constant([[-4, -5, 2],[10, 6, 1],[3, -9, 8]])

A1 = A * (B + C)
A2 = (A * B) + (A * C)

print(A1)
print(A2)

D1 = A * (B * C)
D2 = (A * B) * C

print(D1)
print(D2)

print(tf.equal(A1, A2))

# tf.equal() returns a tensor of the same size. But, to compare the
# values, we can reduce the tensor across all dimensions
if tf.reduce_all(tf.equal(A1, A2)):
    print('Associative property validated')

if tf.reduce_all(tf.equal(D1, D2)):
    print('Distributive property validated')
