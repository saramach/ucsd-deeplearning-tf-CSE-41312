'''
Assignment1 - Problem 5
'''
import tensorflow as tf

a=tf.Variable(1.12)
b=tf.Variable(2.34)
c=tf.Variable(0.72)
d=tf.Variable(0.81)
f=tf.Variable(19.83)

x = 1 + (a/b) + (c/pow(f,2))
s = (b-a)/(d-c)
r = 1/((1/a) + (1/b) + (1/c) + (1/d))
y = a * b * (1/c) * (pow(f,2)/2)

print(x)
print(s)
print(r)
print(y)
