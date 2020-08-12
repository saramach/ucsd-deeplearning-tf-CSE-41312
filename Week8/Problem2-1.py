'''
Week 8 - Problem #2
Cat image - auto-encoder
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import metrics
from PIL import Image
from numpy.random import seed
from tensorflow import set_random_seed

def show_image(img, title):
    fig, aux = plt.subplots(figsize=(5,5))
    aux.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()
    plt.waitforbuttonpress()
    
def read_image():
    img = Image.open('cat.jpg')
    show_image(img, 'actual image')

    # Resize the image
    img.load()
    img = img.resize((128, 128), Image.ANTIALIAS)
    img_array = np.asarray(img)
    img_array = img_array.flatten()
    img_array = np.array([img_array])
    img_array = img_array.astype(np.float32)
    print(img_array)
    return (img_array)
    #return img

def define_model(hidden_layers, image_data):
    # define the keras model
    seed(1)
    set_random_seed(2)
    model = Sequential()
    # Add a hidden layer
    model.add(Dense(hidden_layers, input_dim=image_data.shape[1], activation='relu'))
    model.add(Dense(image_data.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def train_and_predict_model(model, image_data):
    seed(1)
    set_random_seed(2)
    model.fit(image_data, image_data, verbose = 0, epochs=200)
    predict = model.predict(image_data)
    restored_img_data = predict[0].reshape(128, 128, 3)
    restored_img_data = restored_img_data.astype(np.uint8)
    restored_img = Image.fromarray(restored_img_data, 'RGB')
    return restored_img

# Get the image data
image_data = read_image()

# model with 4 input layers
model1 = define_model(4, image_data);
restored_img = train_and_predict_model(model1, image_data)
show_image(restored_img, 'restored image - 4 layers')

# model with 8 input layers
model2 = define_model(8, image_data);
restored_img = train_and_predict_model(model2, image_data)
show_image(restored_img, 'restored image - 8 layers')

# model with 12 input layers
model3 = define_model(12, image_data);
restored_img = train_and_predict_model(model3, image_data)
show_image(restored_img, 'restored image - 12 layers')

# model with 20 input layers
model4 = define_model(20, image_data);
restored_img = train_and_predict_model(model4, image_data)
show_image(restored_img, 'restored image - 20 layers')
