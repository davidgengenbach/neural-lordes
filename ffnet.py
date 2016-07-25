import numpy as np
from perceptron import Perceptron

class ffnet:
    def __init__(self, inpsize,outpsize,hiddenlayers,hiddenlayerheight):
        self.layers = []
        #input layer
        self.layers.append([Perceptron(1,True) for x in range(inpsize)])
        nextinpsize=inpsize

        for l in range(hiddenlayers):
            # hidden layer
            self.layers.append([Perceptron(nextinpsize,False) for x in range(hiddenlayerheight)])
            nextinpsize=hiddenlayerheight

        # output layer
        self.layers.append([Perceptron(nextinpsize,False) for x in range(outpsize)])

    def get_layer_output(self, layer):
        return map(lambda x: x.output, self.layers[layer])

    def propagate(self,inp):
        for index, layer in enumerate(self.layers):
            for per in layer:
                per.propagate(inp)
            inp = self.get_layer_output(index)
        return inp

    def learn(self,opimal_out,alpha):
        for lx in range(len(self.layers)):
            l=len(self.layers)-lx-1
            for p in range(len(self.layers[l])):
                perceptron = self.layers[l][p]
                if lx == 0:
                    perceptron.learn(opimal_out[p], alpha, None)#
                else:
                    trainoutp = 0
                    for p2 in range(len(self.layers[l+1])):
                        trainoutp=trainoutp+self.layers[l+1][p2].delta*self.layers[l+1][p2].weights[p]
                    perceptron.learn(None, alpha, trainoutp)

