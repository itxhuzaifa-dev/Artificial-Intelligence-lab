import numpy as np
import pandas as pd
def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X * w + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Implement Gradient Descent
def gradient_descent(X, y, w, b, alpha, iterations):
    m = len(y)
    cost_history = []  # To store the cost at each iteration

    for _ in range(iterations):
        predictions = X * w + b
        dw = (1 / m) * np.sum((predictions - y) * X)
        db = (1 / m) * np.sum(predictions - y)
        w -= alpha * dw
        b -= alpha * db
        
        # Compute the cost and store it
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

    return w, b, cost_history

def predict(population, w, b):
    return w * population + b