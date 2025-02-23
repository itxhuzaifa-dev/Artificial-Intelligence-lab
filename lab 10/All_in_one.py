# Importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
from utils import *  # Assuming utils.py contains necessary helper functions
import copy
import math

# 2 - Problem Statement
# You are the CEO of a restaurant franchise considering different cities for expansion.
# You have data on existing restaurants' profits and populations, and you want to predict the profit for new cities based on their populations.

# 3 - Dataset
# Loading the dataset
data = pd.read_csv('ex1data1.txt', header=None)
x_train = data[0].values  # Population
y_train = data[1].values  # Profits

# View the variables
print("Type of x_train:", type(x_train))
print("First five elements of x_train are:\n", x_train[:5])

print("Type of y_train:", type(y_train))
print("First five elements of y_train are:\n", y_train[:5])

# Check dimensions
print('The shape of x_train is:', x_train.shape)
print('The shape of y_train is: ', y_train.shape)
print('Number of training examples (m):', len(x_train))

# Visualize the data
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()

# 4 - Refresher on linear regression
# Linear regression model: f(w, b, x) = wx + b

# 5 - Compute Cost
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
    x (ndarray): Shape (m,) Input to the model (Population of cities)
    y (ndarray): Shape (m,) Label (Actual profits for the cities)
    w, b (scalar): Parameters of the model

    Returns:
    total_cost (float): The cost of using w, b as the parameters for linear regression
    """
    m = x.shape[0]
    yp = np.dot(x, w) + b
    total_cost = (1 / (2 * m)) * np.sum((yp - y) ** 2)
    return total_cost

# Testing compute_cost function
initial_w = 2
initial_b = 1
cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(f'Cost at initial w: {cost:.3f}')

# 6 - Gradient descent
def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression

    Args:
    x (ndarray): Shape (m,) Input to the model (Population of cities)
    y (ndarray): Shape (m,) Label (Actual profits for the cities)
    w, b (scalar): Parameters of the model

    Returns:
    dj_dw (scalar): The gradient of the cost w.r.t. the parameter w
    dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    """
    m = x.shape[0]
    yp = np.dot(x, w) + b
    dj_db = (1 / m) * np.sum(yp - y)
    dj_dw = (1 / m) * np.sum((yp - y) * x)
    return dj_dw, dj_db

# Testing compute_gradient function
initial_w = 0
initial_b = 0
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)
print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)

# 2.6 Learning parameters using batch gradient descent
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking num_iters gradient steps with learning rate alpha

    Args:
    x : (ndarray): Shape (m,)
    y : (ndarray): Shape (m,)
    w_in, b_in : (scalar) Initial values of parameters of the model
    cost_function: function to compute cost
    gradient_function: function to compute the gradient
    alpha : (float) Learning rate
    num_iters : (int) number of iterations to run gradient descent

    Returns:
    w : (ndarray): Shape (1,) Updated values of parameters of the model after running gradient descent
    b : (scalar) Updated value of parameter of the model after running gradient descent
    """
    m = len(x)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)
        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w, b, J_history, w_history

# Run gradient descent
initial_w = 0.
initial_b = 0.
iterations = 1500
alpha = 0.01
w, b, _, _ = gradient_descent(x_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print("w, b found by gradient descent:", w, b)

# Plot the linear fit
m = x_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = w * x_train[i] + b

plt.plot(x_train, predicted, c="b")
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()

# Predictions
predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1 * 10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2 * 10000))
