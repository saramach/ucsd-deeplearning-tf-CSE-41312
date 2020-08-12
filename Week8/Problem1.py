'''
Week 8 - Problem #1
Fibonacci - auto-encoder
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import metrics
from numpy.random import seed
from tensorflow import set_random_seed

def create_dataset():
    # 20 data points
    x = np.array([np.arange(0,20)])
    # Data is the fibonacci series
    fib = [1,1]
    num = 20
    for ctr in np.arange(2, num):
        fib.append(fib[ctr-1] + fib[ctr - 2])
    plt.title('generated data')
    plt.scatter(range(20), fib)
    plt.show()
    plt.waitforbuttonpress()
    plt.close()
    np_array_data = np.array([fib])
    np_array_data.astype(np.float32)
    return (x, np_array_data)

x, y = create_dataset()
print(x)
print(y)
seed(1)
set_random_seed(2)
model = Sequential()
# Hidden layer with 20 inputs
model.add(Dense(6, input_dim=x.shape[1], activation='relu'))
# Output layer is the same as input
model.add(Dense(x.shape[1]))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

#Train the model. The output is the same as the input
model.fit(y, y, verbose = 0, epochs=500)
predict = model.predict(y)
print(predict)

#Calculate the RMSE
score = np.sqrt(metrics.mean_squared_error(predict, y))
print("RMSE score: {}".format(score))

plt.title('model output - 6 hidden nodes - 500 epochs')
plt.scatter(x, y, c='red', s=150)
plt.scatter(x, predict, c = 'blue')
plt.show()
plt.waitforbuttonpress()





