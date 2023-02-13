import numpy as np

class LinearRegression():
    def __init__(self,learning_rate = 0.01, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        num_elements = X.shape[0]

        for i in range(self.iterations):
            f_wx_b = np.dot(X, self.w) + self.b #predictions

            dw = (1/num_elements) * (np.dot(X.T, (f_wx_b - y)))
            db = (1/num_elements) * (np.sum(f_wx_b - y))

            self.w = self.w - self.learning_rate*dw
            self.b = self.b - self.learning_rate*db

    def predict(self, X):
        return np.dot(X,self.w) + self.b

    def mean_squared_error(self,y_predictions, y_true):
        squared_error = (y_predictions - y_true)**2
        mse = np.mean(squared_error)
        return mse






