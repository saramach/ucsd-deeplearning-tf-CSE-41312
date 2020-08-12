'''
Week#7 - Problem #1
LSTM - next number in series predictor
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential

#Using TensorFlow backend.
from keras.layers import LSTM

def create_dataset():
    #Create 200 series of numbers, 8 in each
    data_set = [[ [i+j] for i in range(8)] for j in range (200)]
    print('Created dataset...')
    print(data_set[0:3])
    print(data_set[-3:]) 
    
    #Create 200 target, one for each series created earlier
    target_set = [(i+8) for i in range(200)]
    print('Created target...')
    print(target_set[0:3])
    print(target_set[-3:]) 
    np_ds = np.array(data_set, dtype=float)
    np_target = np.array(target_set, dtype = float)

    #Scale it so that the model trains accurately.
    return np_ds/200, np_target/200

def create_train_test_set(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2,
                                                        random_state=4)
    return x_train, x_test, y_train, y_test

def create_train_model(series_train, series_test, target_train, target_test):
    model = Sequential()

    # Add the LSTM
    model.add(LSTM((1), batch_input_shape=(None,8,1), return_sequences=False))
    #model.add(LSTM((1), return_sequences=False))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

    # Dump model parameters
    model.summary()

    # Train the model
    history = model.fit(series_train, target_train, epochs=800, validation_data=(series_test, target_test), verbose=0)

    results = model.predict(series_test)
    plt.title('normalized results over test data')
    plt.scatter(range(40), results, c='r')
    plt.scatter(range(40), target_test, c='g')
    plt.waitforbuttonpress()
    plt.close()

    # Plot the loss Function
    plt.title('loss function')
    plt.plot(history.history['loss'])
    plt.waitforbuttonpress()
    
#Create the dataset
series, target = create_dataset()
#Split into testing and training sets
series_train, series_test, target_train, target_test = create_train_test_set(series, target)
create_train_model(series_train, series_test, target_train, target_test)
