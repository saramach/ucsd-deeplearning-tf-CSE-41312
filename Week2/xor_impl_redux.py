# Assignment 2
# XOR implementation in Tensorflow

# We'll use Tensor flow 1.x
import tensorflow.compat.v1 as tf
import numpy as np

# Disable 2.0 behavior
tf.disable_v2_behavior()

# Input data
x_data = np.array([ [0,0],[1,0],[0,1],[1,1] ])
# Expected Output
y_data = np.array([ [0],[1],[1],[0] ])

learning_rate = 0.1
# Number of iterations
epochs = 10000

# Neural network model parameters
n_input = 2
n_hidden = 3
n_output = 1
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
modelOutput = tf.placeholder(tf.float32)

# Weights from the input layer to the hidden layer
# W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
W1 = tf.Variable([[-4., -6., -5.],[3. , 6., 4.]], name='W1')
# Weights from the hidden layer to the output layer
# W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))
W2 = tf.Variable([[5.],[-9.],[7.]], name='W2')

# Bias values for nodes in hidden layer
# b1 = tf.Variable(tf.zeros([n_hidden]), name='Bias1')
b1 = tf.Variable([-2., 3., -2.], name='Bias1')
# Bias value for the node in the output layer
b2 = tf.Variable([4.], name='Bias2')

# Output of the hidden layer
L2 = tf.sigmoid(tf.matmul(X,W1) + b1)
# Final model output
compOutput = tf.sigmoid(tf.matmul(L2,W2) + b2)

cost = tf.reduce_mean(-Y*tf.log(compOutput) - (1-Y)*tf.log(1-compOutput))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# mean squared error
error = tf.math.square(tf.math.subtract(modelOutput, Y))
init = tf.global_variables_initializer()

# Print helpers
print_W1 = tf.print(W1, summarize=-1)
print_W2 = tf.print(W2, summarize=-1)
print_b1 = tf.print(b1, summarize=-1)
print_b2 = tf.print(b2, summarize=-1)

with tf.Session() as session:
    session.run(init)
    # for step in range(epochs):
    #    session.run(optimizer, feed_dict={X: x_data, Y: y_data})

    # Print what we have
    print('Weights between input layer and hidden layer')
    print('--------------------------------------------')
    session.run(print_W1)

    print('Weights between hidden layer and output layer')
    print('---------------------------------------------')
    session.run(print_W2)

    print('Bias values for the hidden layer')
    print('--------------------------------')
    session.run(print_b1)

    print('Bias value for the output layer')
    print('-------------------------------')
    session.run(print_b2)

    # Compute the value for the four inputs
    print('==========================')
    print('Outputs after the training')
    print('==========================')
    output = session.run(compOutput, feed_dict={X: x_data})
    print(output)

    print('=======================================')
    print('Squared output error after the training')
    print('=======================================')
    print(session.run(error, feed_dict={modelOutput: output, Y: y_data}))
