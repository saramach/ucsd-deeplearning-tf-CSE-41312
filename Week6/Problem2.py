'''
Weekk6 - Problem 2
'''

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image


#Read the image
image = Image.open('02Rice.png')
fig, aux = plt.subplots(figsize=(8,8), num='Original figure')
aux.imshow(image, cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()

#Convert to just the Luminance values
image_lum = image.convert('L')

#Get the image as an array
image_arr = np.asarray(image_lum)

#Laplacian filter #1
lap_filter_1 = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])
conv_image_1 = signal.convolve2d(image_arr, lap_filter_1, mode='same', boundary='symm')
fig, aux = plt.subplots(figsize=(8,8), num='After applying filter 1')
aux.imshow(np.absolute(conv_image_1), cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()
 
#Laplacian filter #2
lap_filter_2 = np.array([[1, 1, 1],
                        [1, -8, 1],
                        [1, 1, 1]])
conv_image_2 = signal.convolve2d(image_arr, lap_filter_2, mode='same', boundary='symm')
#conv_image_2 = signal.convolve2d(conv_image_1, lap_filter_2, mode='same', boundary='symm')
fig, aux = plt.subplots(figsize=(8,8), num='After applying filter 2')
aux.imshow(np.absolute(conv_image_2), cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()

#Laplacian filter #3
lap_filter_3 = np.array([[0, 1, 0],
                        [1, -5, 1],
                        [0, 1, 0]])
conv_image_3 = signal.convolve2d(image_arr, lap_filter_3, mode='same', boundary='symm')
#conv_image_3 = signal.convolve2d(conv_image_2, lap_filter_3, mode='same', boundary='symm')
fig, aux = plt.subplots(figsize=(8,8), num='After applying filter 3')
aux.imshow(np.absolute(conv_image_3), cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()

#Laplacian filter #4
lap_filter_4 = np.array([[1, 1, 1],
                        [1, -9, 1],
                        [1, 1, 1]])
conv_image_4 = signal.convolve2d(image_arr, lap_filter_4, mode='same', boundary='symm')
#conv_image_4 = signal.convolve2d(conv_image_3, lap_filter_4, mode='same', boundary='symm')
fig, aux = plt.subplots(figsize=(8,8), num='After applying filter 4')
aux.imshow(np.absolute(conv_image_4), cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()

#Laplacian filter #1 and #2
conv_image_5 = signal.convolve2d(image_arr, lap_filter_1 + lap_filter_2, mode='same', boundary='symm')
fig, aux = plt.subplots(figsize=(8,8), num='After applying filter 1 + 2')
aux.imshow(np.absolute(conv_image_5), cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()

#Laplacian filter #3 and #4
conv_image_6 = signal.convolve2d(image_arr, lap_filter_3 + lap_filter_4, mode='same', boundary='symm')
fig, aux = plt.subplots(figsize=(8,8), num='After applying filter 3 + 4')
aux.imshow(np.absolute(conv_image_6), cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()

#Laplacian filter #1, #2, #3 and #4
conv_image_7 = signal.convolve2d(image_arr, lap_filter_1 + lap_filter_2 + lap_filter_3 + lap_filter_4, mode='same', boundary='symm')
fig, aux = plt.subplots(figsize=(8,8), num='After applying filter 1 + 2 + 3 + 4')
aux.imshow(np.absolute(conv_image_7), cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()

