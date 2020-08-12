'''
Week6 - Problem 3
Applying Laplacian, Sobel and Canny filters on an image
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2


#Read the image
image = cv2.imread('02Rice.png', 0)
fig, aux = plt.subplots(figsize=(4,4), num='Original figure')
aux.imshow(image, cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()

#Laplacian filter 
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=5)
fig, aux = plt.subplots(figsize=(4,4), num='After applying Laplacian filter with size 5')
aux.imshow(laplacian, cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()

#Sobel-x filter 
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
fig, aux = plt.subplots(figsize=(4,4), num='After applying Sobel filter [X - axis] with size 5')
aux.imshow(sobelx, cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()

#Sobel-y filter 
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
fig, aux = plt.subplots(figsize=(4,4), num='After applying Sobel filter [Y - axis] with size 5')
aux.imshow(sobely, cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()

#Canny edge filter 
canny = cv2.Canny(image, 50,200)
fig, aux = plt.subplots(figsize=(4,4), num='After applying Canny edge detector filte')
aux.imshow(canny, cmap='gray')
plt.show()
plt.waitforbuttonpress()
plt.close()

