'''
Week4 - Assignment problem #1

'''

import math
import matplotlib.pyplot as plt

x = 5
y = 5

learning_rate = 0.01
epsilon = 0.000001
iteration = 0

'''
Function is:
 f(x,y) = z = -1 * math.sqrt(25 - (x - 2) ** 2 - (y - 3) **2)
'''
def func_z(x, y):
    return (-1 * math.sqrt(25 - (x - 2) ** 2 - (y - 3) **2))

'''
Partial derivative w.r.t x
'''
def dz_dx(x, y):
    return ((2 * (x - 2)) / (math.sqrt((25 - (x - 2) ** 2 - (y - 3) ** 2))))

'''
Partial derivative w.r.t y
'''
def dz_dy(x, y):
    return ((2 * (y - 3)) / (math.sqrt((25 - (x - 2) ** 2 - (y - 3) ** 2))))

while True:
    iteration = iteration + 1
    plt.plot(x, y, 'o')
    new_x = x - learning_rate * dz_dx(x,y)
    new_y = y - learning_rate * dz_dy(x,y)
    if (abs(x - new_x) < epsilon) and (abs(y - new_y) < epsilon):
        print("Solution reached..")
        x = new_x
        y = new_y
        plt.plot(x, y, 'o')
        plt.waitforbuttonpress()
        print('New x and y less than epsilon')
        break
    # More improvements could be made
    x = new_x
    y = new_y

print('Solution reached after ' + str(iteration) + ' adjustments')
print('Value of x is: ' + str(x))
print('Value of y is: ' + str(y))
print('Value of z is: ' + str(func_z(x, y)))





