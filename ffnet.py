from perceptron import Perceptron

class ffnet:
    def __init__(self, inpsize,outpsize,hiddenlayers,hiddenlayerheight, activation_function):
        self.arguments = [inpsize,outpsize,hiddenlayers,hiddenlayerheight]
        self.layers = []
        #input layer
        #self.layers.append([Perceptron(1,True) for x in range(inpsize)])
        nextinpsize=inpsize
        self.activation_function = activation_function

        for l in range(hiddenlayers):
            # hidden layer
            self.layers.append([Perceptron(nextinpsize,False, activation_function) for x in range(hiddenlayerheight)])
            nextinpsize=hiddenlayerheight

        # output layer
        self.layers.append([Perceptron(nextinpsize,False, activation_function) for x in range(outpsize)])

    def get_layer_output(self,layer):
        return [self.layers[layer][p].output for p in range(len(self.layers[layer]))]

    def propagate(self,inp):
        for l in range(len(self.layers)):
            for p in range(len(self.layers[l])):
                self.layers[l][p].propagate(inp)
            inp=self.get_layer_output(l)
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