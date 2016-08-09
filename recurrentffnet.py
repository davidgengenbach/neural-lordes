from perceptron import Perceptron
import numpy as np

weightrange=0.8

class recurrentffnet:

    def __init__(self, layers, activation_function):
        self.arguments = [layers]
        self.layers = []
        self.input_size = layers[0]

        nextinpsize = self.input_size
        self.activation_function = activation_function

        for layer in layers[1:-1]:
            self.layers.append([Perceptron(nextinpsize+layer, False, activation_function,random_from=-weightrange, random_to=weightrange) for x in range(layer)])
            nextinpsize = layer

        # output layer
        self.layers.append([Perceptron(nextinpsize, False, activation_function,random_from=-weightrange, random_to=weightrange) for x in range(layers[-1])])


    def get_layer_output(self, layer):
        return [self.layers[layer][p].output for p in range(len(self.layers[layer]))]

    def propagate(self, inp):
        for l in range(len(self.layers)):
            if l!=len(self.layers)-1:
                inp = np.concatenate((inp, self.get_layer_output(l)))
            for p in range(len(self.layers[l])):
                self.layers[l][p].propagate(inp)
            inp = self.get_layer_output(l)
        return inp

    def learn(self, opimal_out, alpha):
        for lx in range(len(self.layers)):
            l = len(self.layers)-lx-1
            for p in range(len(self.layers[l])):
                perceptron = self.layers[l][p]
                if lx == 0:
                    perceptron.learn(opimal_out[p], alpha, None)
                else:
                    trainoutp = 0
                    for p2 in range(len(self.layers[l+1])):
                        trainoutp = trainoutp+self.layers[l+1][p2].delta*self.layers[l+1][p2].weights[p]
                    perceptron.learn(None, alpha, trainoutp)


