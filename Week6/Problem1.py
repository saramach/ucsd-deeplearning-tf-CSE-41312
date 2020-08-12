'''
Week6 - Problem #1
2D Convolution example
'''

from scipy import signal as sg
import tensorflow as tf

'''
Convolution using SciPy
'''
def convolution_scipy():
    image = [[97, 52, 99, 62, 69, 45, 70], 
             [99, 14, 60, 50, 74, 45, 22], 
             [59, 72, 74, 14, 74, 100, 28], 
             [28, 8, 47, 85, 2, 88, 77], 
             [74, 6, 30, 87, 49, 22, 43], 
             [86, 87, 4, 53, 36, 10, 46], 
             [54, 7, 67, 23, 29,26, 15]]

    gaussian_filter = [[1, 2, 1], 
                       [2, 4, 2], 
                       [1, 2, 1]]

    conv_full_result = sg.convolve(image, gaussian_filter)
    conv_valid_result = sg.convolve(image, gaussian_filter,"valid")

    print("Full Convolution result:")
    print(conv_full_result)

    print("Valid Convolution result:")
    print(conv_valid_result)

'''
Convolution usinf Tensorflow
'''
def convolution_tf():
    image_list_tf = tf.constant([[97., 52., 99., 62., 69., 45., 70.], 
                                 [99., 14., 60., 50., 74., 45., 22.], 
                                 [59., 72., 74., 14., 74., 100., 28.], 
                                 [28., 8., 47., 85., 2., 88., 77.], 
                                 [74., 6., 30., 87., 49., 22., 43.], 
                                 [86., 87., 4., 53., 36., 10., 46.], 
                                 [54., 7., 67., 23., 29., 26., 15.]])
    gaussian_filter_list_tf = tf.constant([[1., 2., 1.], 
                                            [2., 4., 2.], 
                                            [1., 2., 1.]])
    image_tf = tf.reshape(image_list_tf, [1,7,7,1])
    gaussian_filter_tf = tf.reshape(gaussian_filter_list_tf, [3, 3, 1, 1])

    conv_full_result = tf.nn.conv2d(image_tf, gaussian_filter_tf, strides=[1,1,1,1], padding='SAME')
    conv_valid_result = tf.nn.conv2d(image_tf, gaussian_filter_tf, strides=[1,1,1,1], padding='VALID')

    with tf.Session() as sess:
        print(conv_full_result.eval())
        print(conv_valid_result.eval())


convolution_scipy()
convolution_tf()

