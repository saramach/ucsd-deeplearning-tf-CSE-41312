'''
Week 3 - Problem 4
Solution using Keras

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing

# Extract data from the CSV file
def extract_predictor_target():
    data = pd.read_csv('kc_house_data.csv')
    print(data.head())
    predictors = preprocessing.minmax_scale(data[['bedrooms','sqft_living']])
    target = preprocessing.minmax_scale(data[['price']])
    #plot_data(predictors, target)
    print(predictors.ndim)
    print(target.ndim)
    return (predictors, target)

# Keras model definition and execution
def model_keras(x_data, y_data, epochs):
    model = Sequential()
    model.add(Dense(1, input_dim=2, kernel_initializer='normal', activation='linear'))
    
    #Compile the model
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])

    #Dump the model
    model.summary()

    #Suppressing the per-epoch messages
    hist = model.fit(x_data, y_data, epochs=epochs, verbose=0)

    weightBias = model.layers[0].get_weights()
    return weightBias

predictors, target = extract_predictor_target()
print('Predictor shape ', predictors.shape)
print('Target shape ', target.shape)

#Convert to arrays
pred_array = np.array(predictors)
target_array = np.array(target)
weightBias = model_keras(pred_array, target_array, 100)
print("RESULT:" + "\n" + "-------")
print("w1: " + str(weightBias[0][0]) + " w2: " + str(weightBias[0][1]) + " bias: " + str(weightBias[1]))
