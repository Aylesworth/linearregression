import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

# Preparing data
data = pd.read_csv('Fish.csv')
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
Xbar = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
Xbar_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

# Solution using gradient descent


# Solution using normal equation
w = np.dot(np.linalg.pinv(np.dot(Xbar.T, Xbar)), np.dot(Xbar.T, y_train))
pred = np.dot(Xbar_test, w)
print(r2_score(y_test, pred))
print(w)

# Solution using scikit-learn
regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))
print(np.concatenate((np.array([regr.intercept_]), regr.coef_)))
