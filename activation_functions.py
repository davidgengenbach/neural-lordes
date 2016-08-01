import numpy as np

class ActivationFunctions:
    @staticmethod
    def soft_plus(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def sigmoid(x):
        return  1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def soft_sign(x):
        return x / (1 + np.abs(x))

    @staticmethod
    def sinoid(x):
        return np.sin(x)

    @staticmethod
    def gaussian(x):
        return np.exp(- (x * x))

    @staticmethod
    def logistic(x):
        return 1 / (1 + np.exp(-x))