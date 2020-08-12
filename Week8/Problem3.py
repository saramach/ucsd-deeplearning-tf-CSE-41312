'''
Week8 - Problem 3
Entropy, Cross entropy and KL divergence
'''

import numpy as np

def entropy(data):
    return np.sum(-data * np.log(data))

def x_entropy(data1, data2):
    return np.sum(-data1 * np.log(data2))

def kl_divergence(data1, data2):
    return np.sum(-data1 * np.log(data2/data1))

img1_dist = np.array([35, 36, 45, 95, 24, 15, 6]) / 256
img2_dist = np.array([20, 56, 85, 52, 22, 20, 1]) / 256

print('Entropy of img1 distribution: {}'.format(entropy(img1_dist)))
print('Entropy of img2 distribution: {}'.format(entropy(img2_dist)))
print('Cross Entropy of img1 w.r.t img2 distribution: {}'.format(x_entropy(img1_dist, img2_dist)))
print('Cross Entropy of img2 w.r.t img1 distribution: {}'.format(x_entropy(img2_dist, img1_dist)))
print('KL divergence of img1 w.r.t img2 distribution: {}'.format(kl_divergence(img1_dist, img2_dist)))
print('KL divergence of img2 w.r.t img1 distribution: {}'.format(kl_divergence(img2_dist, img1_dist)))



