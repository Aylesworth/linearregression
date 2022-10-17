import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# Preparing data
data = pd.read_csv('Fish.csv')
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
Xbar = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
Xbar_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

# Solution using gradient descent
m = Xbar.shape[0]
y_mat = np.reshape(y_train, (y_train.shape[0], 1))


def cost(w):
    return 0.5 / m * (np.linalg.norm(y_mat - Xbar.dot(w)) ** 2)


def grad(w):
    g = 1 / m * Xbar.T.dot(Xbar.dot(w) - y_mat)
    return g


def GD_NAG(grad, w_init, eta=0.02, gamma=0.9, max_iter=400):
    print('Running gradient descent...')
    w = [w_init]
    v = [np.zeros_like(w_init)]

    for it in range(max_iter):
        v_new = gamma * v[-1] + eta * grad(w[-1] - gamma * v[-1])
        w_new = w[-1] - v_new
        w.append(w_new)
        v.append(v_new)
        if np.linalg.norm(grad(w_new)) / np.array(w_init).size < 1e-4:
            break
        # print('Epoch {}/{}, Loss: {}\nw = {}'.format(it, max_iter, cost(w_new), w[-1].T))
    return w[-1]


w_init = np.random.random((Xbar.shape[1], 1))
w = GD_NAG(grad, w_init, eta=0.0004, max_iter=int(1e6))
print("Weights solved using gradient descent:\nw = {}".format(w.reshape((w.shape[0],))))

pred = Xbar_test.dot(w)
print("Accuracy score: {} %\n".format(100 * r2_score(y_test, pred)))

# Solution using normal equation
w = np.dot(np.linalg.pinv(np.dot(Xbar.T, Xbar)), np.dot(Xbar.T, y_train))
pred = np.dot(Xbar_test, w)
print("Weights solved using normal equation:\nw = {}".format(w))
print("Accuracy score: {} %\n".format(100 * r2_score(y_test, pred)))

# Solution using scikit-learn
regr = LinearRegression()
regr.fit(X_train, y_train)
print("Weights solved by scikit-learn:\nw = {}".format(np.concatenate((np.array([regr.intercept_]), regr.coef_))))
print("Accuracy score: {} %\n".format(100 * regr.score(X_test, y_test)))
