import numpy as np
import random


class Perceptron:

    def __init__(self, inpsize, id, activation_function, random_from=-0.05, random_to=0.05):
        self.weights = [random.uniform(random_from, random_to) for x in range(inpsize)]
        self.bias = 0
        self.id = id
        self.inactive = False
        self.activationfunction = activation_function
        self.derivation_factor = 0.000000001

    def activationfunction_der(self, x):
        h = self.derivation_factor
        return (self.activationfunction(x+h)-self.activationfunction(x))/h

    def propagate(self, inp):
        self.input = inp
        self.output = self.activationfunction(
            np.dot(inp, self.weights)+self.bias)
        if self.id == True:
            self.output = inp[0]
        if self.inactive == True:
            self.output = 0
        return self.output

    def learn(self, opimal_out, alpha, errorterm):
        if errorterm == None:
            errorterm = self.output-opimal_out
        out_der = self.activationfunction_der(
            np.dot(self.input, self.weights)+self.bias)
        self.delta = out_der * errorterm
        self.weights = self.weights - alpha*self.delta*np.transpose(self.input)
        self.bias = self.bias-alpha*self.delta
