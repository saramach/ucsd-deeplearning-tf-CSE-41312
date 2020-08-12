'''
Week 3 - Problem 2
Solution using Keras

'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

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

def model_keras(x_data, y_data, epochs):
    model = Sequential()
    model.add(Dense(1, input_dim=1, kernel_initializer='normal', activation='linear'))
    
    #Compile the model
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])

    #Dump the model
    model.summary()

    #Suppressing the per-epoch messages
    hist = model.fit(x_data, y_data, epochs=epochs, verbose=0)

    weightBias = model.layers[0].get_weights()
    #print('Weight and Bias with Keras: " +  weightBias)
    print(weightBias)
    plt.plot(train_x, train_y,'o')
    plt.plot(x_data, weightBias[0][0]*x_data + weightBias[1])
    plt.waitforbuttonpress()

train_x, train_y = generate_data(42, 30)
model_keras(train_x, train_y, 20000)
