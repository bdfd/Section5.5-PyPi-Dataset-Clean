'''
Date         : 2023-12-01 11:43:34
Author       : BDFD,bdfd2005@gmail.com
Github       : https://github.com/bdfd
LastEditTime : 2023-12-01 11:43:53
LastEditors  : BDFD
Description  : 
FilePath     : \test.py
Copyright (c) 2023 by BDFD, All Rights Reserved. 
'''
# Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generating some random data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Creating random input features
y = 3 + 4 * X + np.random.randn(100, 1)  # Generating output with some noise

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predicting values
X_new = np.array([[0], [2]])  # New input data
y_pred = model.predict(X_new)  # Predicting output for new data

# Plotting the data and the linear regression line
plt.scatter(X, y, label='Actual data')
plt.plot(X_new, y_pred, 'r-', label='Predicted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()
