import numpy as np
import matplotlib.pyplot as plt

class Regression():
    def __init__(self, learning_rate = 0.01, iterations = 1000):
        self.lr_rate = learning_rate
        self.iter = iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        num_elements = X.shape[0]

        for i in range(self.iter):
            f_wx_b = np.dot(X, self.w) + self.b

            dw = (1/num_elements) * np.dot(X.T, (f_wx_b - y))
            db = (1/num_elements) * np.sum((f_wx_b - y))
            self.w = self.w - self.lr_rate*dw
            self.b = self.b - self.lr_rate*db

class LinearRegression(Regression):
    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def mean_squared_error(self, y_predictions, y_true):
        squared_error = (y_predictions - y_true)**2
        mse = np.mean(squared_error)
        return mse

    def plot(self, X, y):
        y_pred = self.predict(X)
        plt.scatter(y, y_pred, s=10)
        plt.plot(y, y, color='r')
        plt.show()


class LogisticRegression(Regression):
    def predict(self, X):
        f_wx_b = np.dot(X, self.w) + self.b
        results = self.sigmoid(f_wx_b)
        class_pred = []
        for i in results:
            if y <= 0.5:
                class_pred.append(0)
            else:
                class_pred.append(1)
        class_pred = np.array(class_pred)
        return class_pred

    def sigmoid(self, fwxb):
        return 1/(1+np.exp(-fwxb))

    def binary_cross_entropy(self, y_predictions, y_true):
        epsilon = 1e-15
        y_predictions = np.clip(y_predictions, epsilon, 1 - epsilon)
        return np.mean(-(y_true*np.log(y_predictions) + (1-y_true)*np.log(1-y_predictions)))

