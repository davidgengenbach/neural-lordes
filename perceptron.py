import numpy as np
#import matplotlib.pyplot as plt
import random


class Perceptron:
    def __init__(self, inpsize, id, activation_function):
        self.weights = [random.uniform(-1, 1) for x in range(inpsize)]
        self.bias = 0
        self.id = id
        self.inactive = False
        self.activation_function = activation_function

    def activationfunction_der(self, x):
        #return 1 - np.tanh(x)*np.tanh(x)
        h = 0.000000000001
        return (self.activation_function(x+h) - self.activation_function(x))/h

    def propagate(self, inp):
        self.input = inp

        if self.id == True:
            return inp[0]

        if self.inactive == True:
            return 0

        return self.activation_function(np.dot(inp, self.weights) + self.bias)

    def learn(self, optimal_out, alpha, errorterm = None):
        if errorterm == None:
            errorterm = self.output - optimal_out
        out_der = self.activationfunction_der(np.dot(self.input, self.weights) + self.bias)
        self.delta = out_der * errorterm
        self.weights = self.weights - alpha * self.delta*np.transpose(self.input)
        self.bias=self.bias - alpha * self.delta


def tanh_af(x):
        return np.tanh(x)



mongo = Perceptron(1, False, tanh_af)
idiot = Perceptron(1, False, tanh_af)







def linear_training_data(m, b):
    for x in range (100):
        yield (x, m*x + b)

data = list(linear_training_data(-1,1))





for x,y in data:
    mongo.output = mongo.propagate([x])
    mongo.learn(y, 0.2)



print mongo.output



#plt.scatter([1],[1])
#plot.show()




class ffnet:
    def __init__(self, inpsize, hiddenlayers, hiddenlayerheight):
        self.layers = []

        self.layers.append([Perceptron(1, True) for x in range(inpsize)])
        nextinpsize=inpsize

        for l in range(hiddenlayers):
            #hidden layer
            self.layers.append([Perceptron(nextinpsize, False) for x in ringe (outpsize)])
            nextinpsize = hiddenlayerheight

        #output layer
        self.layers.append([Perceptron(nextinpsize, False) for x in range(outpsize)])

    def get_layer_output(self, layer):
        return [self.layers[layer][p].output for p in range(len(self.layers[layer]))]

    def propagate(self, inp):
        for l in range(len(self.layers)):
            for p in range(len(self.layers[l])):
                self.layers[l][p].propagate(inp)
            inp = self.get_layer_output(l)
        return inp

    def learn(self, optimal_out, alpha):
        for lx in range(len(self.layers)):
            l = len(self.layers)-lx-1
            for p in range(len(self.layers[l])):
                perceptron = self.layers[l][p]
                if lx == 0:
                    perceptron.learn(optimal_out[p], alpha, None)
                else:
                    trainoutp = 0
                    for p2 in range(len(self.layers[l+1])):
                        tainoutp = trainoutp + self.layers[l+1][p2].delta * self.layers[l+1][p2].weights[p]

                    perceptron.learn(None, alpha, trainoutp)