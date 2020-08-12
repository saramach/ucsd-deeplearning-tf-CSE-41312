'''
Week 3 - Problem 3
Solution using Scikit-learn and Tensorflow
Housing data in a CSV file
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow.compat.v1 as tf
from sklearn import linear_model
from sklearn import preprocessing
import pandas as pd

# Disable 2.0 behavior
tf.disable_v2_behavior()

def plot_plane(predictors, target, b1, b2, intercept):
    plt.clf()
    figure = plt.figure()
    x1_surf, x2_surf = np.meshgrid(np.linspace(predictors[:,0].min(), predictors[:,0].max(), 500), np.linspace(predictors[:,1].min(), predictors[:,1].max(), 500))
    y_surf = x1_surf*b1 + x2_surf*b2 + intercept
    ax = Axes3D(figure)
    ax.scatter(predictors[:,0], predictors[:,1], target, c='blue', alpha=0.5)
    plt_surface = ax.plot_surface(x1_surf, x2_surf, y_surf, color='red', alpha=0.2)
    ax.set_xlabel('Bedrooms')
    ax.set_ylabel('Sqft')
    ax.set_zlabel('Price')
    plt.show()
    plt.waitforbuttonpress()

### Import the data from the CSV file ###
def extract_predictor_target():
    data = pd.read_csv('kc_house_data.csv')
    print(data.head())
    predictors = preprocessing.minmax_scale(data[['bedrooms','sqft_living']])
    target = preprocessing.minmax_scale(data[['price']])
    #plot_data(predictors, target)
    print(predictors.ndim)
    print(target.ndim)
    return (predictors, target)

### SciKit Learn method
def scikit_method(predictors, target):
    print("Using SciKit learn..")
    linear_reg = linear_model.LinearRegression()
    #print("Dimensions: X: " + str(predictors.ndim) + ", Y: " + str(target.ndim))
    # IMPORTANT: LinearRegression expects a 2-D array. So, add a dimension using
    # reshape()
    #linear_reg.fit(x_data.reshape(-1,1), y_data.reshape(-1,1))
    linear_reg.fit(predictors, target)
    print("Slope : " + str(linear_reg.coef_))
    print("Intercept: " + str(linear_reg.intercept_))
    return (linear_reg.coef_, linear_reg.intercept_)

### TensorFlow method
def tensorflow_method(x_data, y_data, learn_rate, epochs):
    print("Tensor flow method..")
    graph = tf.Graph()
    with graph.as_default():
        slope = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        x1 = tf.placeholder(dtype=np.float32)
        x2 = tf.placeholder(dtype=np.float32)
        y = tf.placeholder(dtype=np.float32)

        w1 = tf.Variable([0], dtype=np.float32, name="weight1")
        w2 = tf.Variable([0], dtype=np.float32, name="weight2")
        b = tf.Variable([0], dtype=np.float32, name="bias")

        response = w1*x1 + w2*x2 + b
        cost = tf.reduce_mean(tf.square(response - y))
        optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

        with tf.Session(graph=graph) as session:
            init = tf.global_variables_initializer()
            session.run(init)
            
            for epoch in range(epochs):
                session.run(optimizer, {x1:x_data[:,0].tolist(), x2:x_data[:,1].tolist(), y:y_data[:,0].tolist()})
                if ( epoch % 10000 ) == 0:
                    print("w1 = ",session.run(w1))
                    print("w2 = ",session.run(w2))
                    print("b = ",session.run(b))

            return(session.run(w1), session.run(w2), session.run(b))

predictors, target = extract_predictor_target()
print(predictors)
print(target)
print(predictors[:,0], predictors[:,1], target[:,0])

#Get the linear regression using SciKit Learn
coef, intercept = scikit_method(predictors, target)
print(coef[0][0])
print(coef[0][1])
print(intercept[0])
plot_plane(predictors, target, coef[0][0], coef[0][1], intercept[0])
w1, w2, b = tensorflow_method(predictors, target, 0.000001, 100000)
#plot_plane(predictors, target, w1, w2, b)

print("RESULTS" + "\n" + "-------")
print("SciKit Learn : w1: " + str(coef[0][0]) + " w2: " + str(coef[0][1]) + " b: " + str(intercept[0]))
print("Tensorflow: w1: " + str(w1) + " w2: " + str(w2) + " b: " + str(b))

