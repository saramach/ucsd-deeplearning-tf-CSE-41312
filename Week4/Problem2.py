'''
Week 4 - Problem 2
Solution using Scikit-learn and hand-coded Gradient descent method

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing

### Generate the data ###
def generate_data(random_seed, n_samples):
    train_x = np.linspace(0,20,n_samples)
    train_y = 3.7 * train_x + 14 + 4 * np.random.randn(n_samples)
    print("X data")
    print("------")
    print("Size: " + str(np.shape(train_x)))
    print(train_x)
    print("Y data")
    print("------")
    print("Size: " + str(np.shape(train_y)))
    print(train_y)
    return(train_x, train_y)

### SciKit Learn method
def scikit_method(x_data, y_data):
    print("Using SciKit learn..")
    linear_reg = linear_model.LinearRegression()
    print("Dimensions: X: " + str(x_data.ndim) + ", Y: " + str(y_data.ndim))
    # IMPORTANT: LinearRegression expects a 2-D array. So, add a dimension using
    # reshape()
    linear_reg.fit(x_data.reshape(-1,1), y_data.reshape(-1,1))
    print("Slope : " + str(linear_reg.coef_))
    print("Intercept: " + str(linear_reg.intercept_))
    return (linear_reg.coef_, linear_reg.intercept_)
    
### Handcoded Gradient descent method
'''
Partial differentiation w.r.t slope
'''
def dy_dslope(slope, intercept, x_data, y_data):
    return (-2 * sum((y_data - slope * x_data - intercept) * x_data))

'''
Partial differentiation w.r.t intercept
'''
def dy_dintercept(slope, intercept, x_data, y_data):
    return (-2 * sum((y_data - slope * x_data - intercept)))

'''
Gradient Descent method
'''
def gradient_descent(x_data, y_data, learn_rate, epochs):
    print("Gradient Descent method..")
    slope = 0
    intercept = 0

    for iteration in range(epochs):
        #print(iteration)
        slope = slope - learn_rate * dy_dslope(slope, intercept, x_data, y_data)
        intercept = intercept - learn_rate * dy_dintercept(slope, intercept, x_data, y_data)
    
    return (slope, intercept)

'''
Function to descale a min_max scaled data
'''
def deScale_y(y_orig, scaled_y):
    result_y = []
    min_y = min(y_orig)
    max_y = max(y_orig)
    for each_scaled_y in scaled_y:
        result_y.append(each_scaled_y * (max_y - min_y) + min_y)

    return result_y

    
train_x_actual, train_y_actual = generate_data(42, 30)
train_x = preprocessing.minmax_scale(train_x_actual)
train_y = preprocessing.minmax_scale(train_y_actual)

s_slope, s_intercept = scikit_method(train_x, train_y)
gd_slope, gd_intercept = gradient_descent(train_x, train_y, 0.001, 4000)

print("RESULTS" + "\n" + "-------")
print("SciKit Learn : slope: " + str(s_slope) + " intercept: " + str(s_intercept))
sci_slope = s_slope[0][0]
sci_intercept = s_intercept[0]
result_scaled_y = train_x * sci_slope + sci_intercept
sci_descaled_computed_y = deScale_y(train_y_actual, result_scaled_y)

print("Gradient Descent : slope: " + str(gd_slope) + " intercept: " + str(gd_intercept))
result_scaled_y = train_x * gd_slope + gd_intercept
gd_descaled_computed_y = deScale_y(train_y_actual, result_scaled_y)

# Plot the results
plt.plot(train_x_actual, train_y_actual, 'o')
plt.plot(train_x_actual, sci_descaled_computed_y, '-o')
plt.plot(train_x_actual, gd_descaled_computed_y, '-o')
plt.show()
plt.waitforbuttonpress()




