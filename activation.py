import numpy as np


class Activation():
    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        """
        Derivative of sigmoid function
        """
        return np.exp(-z)/((1 + np.exp(-z))**2)