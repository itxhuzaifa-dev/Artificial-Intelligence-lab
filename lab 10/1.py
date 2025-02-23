import numpy as np
import matplotlib.pyplot as plt
from utils import  gradient_descent, predict
import pandas as pd
 
# Load the dataset
data = pd.read_csv('ex1data1.txt', header=None)
X = data[0].values  # Population
y = data[1].values  # Profits

# Visualize Data
plt.scatter(X, y, color='red', marker='x')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Population vs. Profit')
plt.show()

# Implement Cost Function

# Initialize parameters
w = 0
b = 0
alpha = 0.01  # Learning rate
iterations = 1000

# Fit the Model
w, b, cost_history = gradient_descent(X, y, w, b, alpha, iterations)

# Plot the cost history
plt.plot(range(iterations), cost_history, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Reduction Over Time')
plt.show()

# Make Predictions

population_35000 = 35000
population_70000 = 70000
print(f'Predicted profit for 35,000 population: ${predict(population_35000, w, b):,.2f}')
print(f'Predicted profit for 70,000 population: ${predict(population_70000, w, b):,.2f}')
