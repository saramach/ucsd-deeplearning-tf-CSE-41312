'''
Week 3 - Problem 1
Solution using Scikit-learn and Tensorflow

'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn import linear_model

# Disable 2.0 behavior
tf.disable_v2_behavior()

### Generate the data ###
def generate_data(random_seed, n_samples):
    tf.set_random_seed(random_seed)
    train_x = np.linspace(0,20,n_samples)
    train_y = 3.7 * train_x + 14 + 4 * np.random.randn(n_samples)
    print("X data")
    print("------")
    print("Size: " + str(np.shape(train_x)))
    print(train_x)
    print("Y data")
    print("------")
    print("Size: " + str(np.shape(train_y)))
    print(train_y)
    plt.plot(train_x, train_y,'o')
    plt.waitforbuttonpress()
    plt.close()
    return(train_x, train_y)

### SciKit Learn method
def scikit_method(x_data, y_data):
    print("Using SciKit learn..")
    linear_reg = linear_model.LinearRegression()
    print("Dimensions: X: " + str(x_data.ndim) + ", Y: " + str(y_data.ndim))
    # IMPORTANT: LinearRegression expects a 2-D array. So, add a dimension using
    # reshape()
    linear_reg.fit(x_data.reshape(-1,1), y_data.reshape(-1,1))
    print("Slope : " + str(linear_reg.coef_))
    print("Intercept: " + str(linear_reg.intercept_))
    return (linear_reg.coef_, linear_reg.intercept_)
    
### TensorFlow method
def tensorflow_method(x_data, y_data, learn_rate, epochs):
    print("Tensor flow method..")
    graph = tf.Graph()
    with graph.as_default():
        slope = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        intercept = tf.Variable(tf.zeros([1]))
        response = slope*x_data + intercept

        cost = tf.reduce_mean(tf.square(response - y_data))
        optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

        with tf.Session(graph=graph) as session:
            init = tf.global_variables_initializer()
            session.run(init)
            
            for epoch in range(epochs):
                session.run(optimizer)
                if ( epoch % 1000 ) == 0:
                    print("Plot after " + str(epoch) + " iterations")
                    plt.plot(x_data, y_data, 'o', label = 'step = {}'.format(epoch))
                    plt.plot(x_data, session.run(slope)*x_data + session.run(intercept))
                    plt.legend()
                    plt.show()
            plt.waitforbuttonpress()
            print("Slope = ",session.run(slope))
            print("Intercept = ",session.run(intercept))

            #return(tf.cast(session.run(slope), tf.int32), tf.cast(session.run(intercept), tf.int32))
            return(session.run(slope), session.run(intercept))

train_x, train_y = generate_data(42, 30)
s_slope, s_intercept = scikit_method(train_x, train_y)
t_slope, t_intercept = tensorflow_method(train_x, train_y, 0.001, 10000)

print("RESULTS" + "\n" + "-------")
print("SciKit Learn : slope: " + str(s_slope) + " intecept: " + str(s_intercept))
print("TensorFlow : slope: " + str(t_slope) + " intecept: " + str(t_intercept))


