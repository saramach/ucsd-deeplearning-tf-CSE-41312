'''
Week5 - Assignment problem #1
Solving Week4's problem using Gradient Descent with Momentum
'''

import math
import matplotlib.pyplot as plt

learning_rate = 0.01
epsilon = 0.000001

'''
Function is:
 f(x,y) = z = -1 * math.sqrt(25 - (x - 2) ** 2 - (y - 3) **2)
'''
def func_z(x, y):
    return (-1 * math.sqrt(25 - (x - 2) ** 2 - (y - 3) **2))

'''
Partial derivative w.r.t x
NOTE: Fixing the derivative from Homework4
'''
def dz_dx(x, y):
    return ((x - 2) / (math.sqrt((25 - (x - 2) ** 2 - (y - 3) ** 2))))

'''
Partial derivative w.r.t y
NOTE: Fixing the derivative from Homework4
'''
def dz_dy(x, y):
    return ((y - 3) / (math.sqrt((25 - (x - 2) ** 2 - (y - 3) ** 2))))

'''
Plain Gradient Descent
'''
def plain_gradient_descent(x, y):
    iteration = 0
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
            plt.close()
            print('New x and y less than epsilon')
            return (x, y, iteration)
        # More improvements could be made
        x = new_x
        y = new_y

'''
Gradient Descent with momentum
'''
def gradient_descent_momentum(x, y):
    update_x = 0
    update_y = 0
    gamma = 0.9
    iteration = 0
    while True:
        iteration = iteration + 1
        plt.plot(x, y, 'o')
        update_x = (gamma * update_x) + (learning_rate * dz_dx(x,y))
        update_y = (gamma * update_y) + (learning_rate * dz_dy(x,y))
         
        new_x = x - update_x
        new_y = y - update_y
        if (abs(x - new_x) < epsilon) and (abs(y - new_y) < epsilon):
            print("Solution reached..")
            x = new_x
            y = new_y
            plt.plot(x, y, 'o')
            plt.waitforbuttonpress()
            plt.close()
            print('New x and y less than epsilon')
            return (x, y, iteration)
        # More improvements could be made
        x = new_x
        y = new_y

    

#First, try using Gradient Descent
x,y,iteration = plain_gradient_descent(5,5)

print('Plain Gradient Descent')
print('Solution reached after ' + str(iteration) + ' adjustments')
print('Value of x is: ' + str(x))
print('Value of y is: ' + str(y))
print('Value of z is: ' + str(func_z(x, y)))

#Try the GD, with momentum
x,y,iteration = gradient_descent_momentum(5,5)
print('Gradient Descent with Momentum')
print('Solution reached after ' + str(iteration) + ' adjustments')
print('Value of x is: ' + str(x))
print('Value of y is: ' + str(y))
print('Value of z is: ' + str(func_z(x, y)))
