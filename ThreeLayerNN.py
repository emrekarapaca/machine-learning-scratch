import numpy as np
#0.9289
#used for MNIST dataset

def get_predictions(a3):
    return np.argmax(a2, 0)


def get_accuracy(predicts, y):
    print(predicts, y)
    return np.sum(predicts == y) / y.size


def relu(z):
    return np.maximum(z, 0)


def softmax(z):
    sm = np.exp(z) / sum(np.exp(z))
    return sm


def relu_deriv(z):
    return z > 0


def encoding(y):
    one_encoding = np.zeros([10, 42000])
    for i in range(y.shape[0]):
        one_encoding[y[i], i] = 1
    return one_encoding


class ThreeLayerNN:
    def __init__(self, iterations=500, learning_rate=0.01, lambda_=0.1):
        self.w3 = None
        self.b3 = None
        self.b2 = None
        self.w2 = None
        self.b1 = None
        self.w1 = None
        self.alpha = learning_rate
        self.iterations = iterations
        self.lambda_ = lambda_
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None

    def init_params(self):
        self.w1 = np.random.rand(10, 784) - 0.5
        self.b1 = np.random.rand(10, 1) - 0.5
        self.w2 = np.random.rand(30, 10) - 0.5
        self.b2 = np.random.rand(30, 1) - 0.5
        self.w3 = np.random.rand(10, 30) - 0.5
        self.b3 = np.random.rand(10, 1) - 0.5

    def forward_prop(self, x):
        self.z1 = np.dot(self.w1, x) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = relu(self.z2)
        self.z3 = np.dot(self.w3, self.a2) + self.b3
        self.a3 = softmax(self.z3)

    def backward_prop(self, x, y):
        num_elements = x.shape[0]
        y_hat = encoding(y)
        dz3 = (self.a3 - y_hat)
        dw3 = (1 / num_elements) * np.dot(dz3, self.a2.T) + (self.lambda_ * self.w3)
        db3 = (1 / num_elements) * np.sum(dz3)
        dz2 = np.dot(self.w3.T, dz3) * relu_deriv(self.z2)
        dw2 = (1 / num_elements) * np.dot(dz2, self.a1.T) + (self.lambda_ * self.w2)
        db2 = (1 / num_elements) * np.sum(dz2)
        dz1 = np.dot(self.w2.T, dz2) * relu_deriv(self.z1)
        dw1 = (1 / num_elements) * np.dot(dz1, x.T) + (self.lambda_ * self.w1)
        db1 = (1 / num_elements) * np.sum(dz1)
        return dw3, db3, dw2, db2, dw1, db1

    def update_params(self, dw1, db1, dw2, db2, dw3, db3):
        self.w1 = self.w1 - self.alpha * dw1
        self.b1 = self.b1 - self.alpha * db1
        self.w2 = self.w2 - self.alpha * dw2
        self.b2 = self.b2 - self.alpha * db2
        self.w3 = self.w3 - self.alpha * dw3
        self.b3 = self.b3 - self.alpha * db3

    def gradient_descent(self, x, y):
        self.init_params()
        for i in range(self.iterations):
            self.forward_prop(x)
            dw3, db3, dw2, db2, dw1, db1 = self.backward_prop(x, y)
            self.update_params(dw1, db1, dw2, db2, dw3, db3)
            if i % 50 == 0:
                print("Iteration: ", i)
                predictions = get_predictions(self.a3)
                print(get_accuracy(predictions, y))
        return self.w1, self.b1, self.w2, self.b2, self.w3, self.b3


if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv('train.csv')
    data = np.array(data)
    X_train = data[:, 1:]
    X_train = X_train.T
    X_train = X_train / 255
    epsilon = 1e-15
    X_train = np.clip(X_train, epsilon, 1 - epsilon)
    y_train = data[:, 0]
    nn = ThreeLayerNN()
    w1, b1, w2, b2, w3, b3 = nn.gradient_descent(X_train, y_train)
