# Assignment 2
# Sales prediction

# We'll use Tensor flow 1.x
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Disable 2.0 behavior
tf.disable_v2_behavior()

# Random seed initialization
RANDOM_SEED = 55
tf.set_random_seed(RANDOM_SEED)

# Input file data
sales_file_data = pd.read_csv('Advertising.csv')
#print(sales_file_data)

# Split features and result
ad_media = sales_file_data[['TV','Radio','Newspaper']]
sales = sales_file_data[['Sales']]
#print(ad_media)
#print(sales)

# Scale the features
# NOTE: We don't need to scale the sales.
scaled_ad_media = preprocessing.minmax_scale(ad_media)
print(scaled_ad_media)

# Need to convert sales from dataFrame to numpy array
# NOTE: This is very important.
sales=sales.to_numpy()
print(sales)

print("Splitting trian and test set")
# Split the input data into training and testing partitions
train_media, test_media, train_sales, test_sales = train_test_split(scaled_ad_media, sales, test_size=0.30, random_state=RANDOM_SEED)

ad_media_shape = train_media.shape[1]
sales_shape = train_sales.shape[1]

learning_rate = 0.008
# Number of iterations
epochs = 3000

# ====================================================
# Neural network model parameters
# Inputs are TV, Radio and Newspaper
n_input = ad_media_shape 
n_hidden = 8
# The output is a single value
n_output = sales_shape

print("model dimenstions: input: {inp}, hidden: {hidden}, output: {out}".format(inp=n_input, hidden=n_hidden, out=n_output))
inputs = tf.placeholder("float", shape=[None, n_input])
output = tf.placeholder("float", shape=[None, n_output])
# ====================================================

# Weights from the input layer to the hidden layer
W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
# weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
# W1 = tf.get_variable(name="Weight1", dtype=tf.float32, shape=[n_input, n_hidden], initializer=weight_initer)
# W1 = tf.get_variable(name="W1", shape=[n_input, n_hidden], initializer=tf.contrib.layers.xavier_initializer())

# Weights from the hidden layer to the output layer
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))
# W2 = tf.get_variable(name="Weight2", dtype=tf.float32, shape=[n_hidden, n_output], initializer=weight_initer)
# W2 = tf.get_variable(name="W1", shape=[n_hidden, n_output], initializer=tf.contrib.layers.xavier_initializer())

# Bias values for nodes in hidden layer
b1 = tf.Variable(tf.zeros([n_hidden]), name='Bias1')
# Bias value for the node in the output layer
b2 = tf.Variable(tf.zeros([n_output]), name='Bias2')

# Use RELU for the activation function
# Output of the hidden layer
L2 = tf.nn.relu(tf.matmul(inputs,W1) + b1)
# Final model output
compOutput = tf.math.add(tf.matmul(L2,W2), b2)

# Linear regression model cost function
cost = tf.reduce_mean(tf.math.square(tf.math.subtract(compOutput, output)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# root mean squared error (RMSE)
modelOutput = tf.placeholder("float", shape=[None, n_output])
error = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.math.subtract(modelOutput, output))))
init = tf.global_variables_initializer()

# Print helpers
print_W1 = tf.print(W1, summarize=-1)
print_W2 = tf.print(W2, summarize=-1)
print_b1 = tf.print(b1, summarize=-1)
print_b2 = tf.print(b2, summarize=-1)

with tf.Session() as session:
    session.run(init)
    print("****** Model training begin ******")
    print("Length of train media:" + str(len(train_media)))
    print("Length of train sales:" + str(len(train_sales)))
    print(train_media[0:1])
    print(train_sales[0:1])
    for step in range(epochs):
        # Train with each example
        # Train the model with the training set
        for i in range(len(train_media)):
            session.run(optimizer, feed_dict={inputs: train_media[i: i + 1], output: train_sales[i: i + 1]})

    print("****** Model training complete ******")
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

    print("Based on the model, this is what the sales will be for the test media input")
    test_output = session.run(compOutput, feed_dict={inputs: test_media})
    print(test_output)

    print("Computing the accuracy")
    test_accuracy = session.run(error, feed_dict={modelOutput: test_output, output: test_sales})

    print(test_media)
    print(test_sales)
    print(test_output)

    print("Test Accuracy:")
    print("==============")
    print(test_accuracy)
