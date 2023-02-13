import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():
    def __init__(self, learning_rate = 0.01, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        num_elements = X.shape[0]

        for i in range(self.iterations):
            f_wx_b = np.dot(X, self.w) + self.b
            y_pred = sigmoid(f_wx_b)

            dw = (1/num_elements)*(np.dot(X.T,(y_pred-y)))
            db = (1/num_elements)*(np.sum(y_pred-y))

            self.w = self.w - self.learning_rate*dw
            self.b = self.b - self.learning_rate*db

    def predict(self, X):
        f_wx_b = np.dot(X, self.w) + self.b
        y_pred = sigmoid(f_wx_b)
        class_pred = []
        for y in y_pred:
            if y <= 0.5:
                class_pred.append(0)
            else:
                class_pred.append(1)
        class_pred = np.array(class_pred)
        return class_pred

    def binary_cross_entropy(self,y_true, y_predictions):
        epsilon = 1e-15
        y_predictions = np.clip(y_predictions, epsilon, 1 - epsilon)
        return np.mean(-(y_true*np.log(y_predictions) + (1-y_true)*np.log(1-y_predictions)))



