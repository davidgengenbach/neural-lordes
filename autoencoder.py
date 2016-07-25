import numpy as np
import matplotlib.pyplot as plt
import random

class Perceptron:
    def __init__(self, inpsize,id):
        self.weights = [random.uniform(-0.01, 0.01) for x in range(inpsize)]
        self.bias=0
        self.id=id
        self.inactive=False

    def activationfunction(self,x):
        #return (1/(1*np.sqrt(2*np.pi))*np.exp(-0.5*np.power(x/1,2))) #gauss
        #return 1 - np.power((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)), 2) #tanh der
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)) #tanh

    def activationfunction_der(self,x):
        h=0.000000001
        return (self.activationfunction(x+h)-self.activationfunction(x))/h

        #return 1-np.power(self.activationfunction(x),2)

    def propagate(self,inp):
        self.input=inp
        self.output=self.activationfunction(np.dot(inp, self.weights)+self.bias)#
        if self.id == True:
            self.output=inp[0]

        if self.inactive==True:
            self.output=0

        return self.output


    def learn(self,opimal_out,alpha,errorterm):
        if errorterm==None:
            errorterm = self.output-opimal_out
        out_der = self.activationfunction_der(np.dot(self.input, self.weights)+self.bias)
        self.delta =  out_der * errorterm
        self.weights = self.weights - alpha*self.delta*np.transpose(self.input)
        self.bias=self.bias-alpha*self.delta

class ffnet:
    def __init__(self, inpsize,outpsize,hiddenlayers,hiddenlayerheight):
        self.layers = []
        #input layer
        #self.layers.append([Perceptron(1,True) for x in range(inpsize)])
        nextinpsize=inpsize

        for l in range(hiddenlayers):
            # hidden layer
            self.layers.append([Perceptron(nextinpsize,False) for x in range(hiddenlayerheight)])
            nextinpsize=hiddenlayerheight

        # output layer
        self.layers.append([Perceptron(nextinpsize,False) for x in range(outpsize)])

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


autoencoder=ffnet(100,100,1,1)

def target(x):
    #return np.power(x,1.1)
    return np.sin(x*8)*0.9


xrange=np.arange(0, 1, 0.01)
input=target(xrange)
#print xrange

#print input

def plotvec(res,c):
    plt.plot(xrange, res, color=c)

plotvec(input,'blue')

for cycle in range(100):
    print cycle
    out=np.array(autoencoder.propagate(input))
    #print out

    plotvec(out,'green')

    autoencoder.learn(input,0.05)

    for p in range(len(autoencoder.layers[1])):
        for w in range(len(autoencoder.layers[1][p].weights)):
            autoencoder.layers[1][p].weights[w]=autoencoder.layers[0][w].weights[p]

out = autoencoder.propagate(input)
plotvec(out, 'red')



plt.show()

'''


#in_data=(1.0)
#out_data=(0.7)

net=ffnet(1,1,2,10)

neuron=Perceptron(1,False)

for i in range(100):
    print net.propagate([1])
    net.learn([0.5],0.5)

def target(x):
    #return np.power(x,1.1)
    return np.sin(x*8)

def netf(x):
    return net.propagate([x])

def netfvec(x):
    return [net.propagate([xi]) for xi in x]

r=np.arange(0, 1, 0.05)
#r2=np.arange(-1, 2, 0.05)
plt.plot(r, target(r))

for i in range(200):
    #print i
    alpha=0.01#random.uniform(0.000001, 0.01)
    #net.randomshutoff(5)
    rran = random.sample(r, len(r))

    for x in rran:
        ty=target(x)
        y = net.propagate([x])
        net.learn([ty], alpha)#0.001
        #plt.scatter([x], [y])

    if i%10==0:
    #    print netfvec(r)
        plt.plot(r, netfvec(r), color='green')

    #for x in r:
    #    plt.scatter([x], net.propagate([x]),color='green')
        #for xt in range(100):
        #    plt.scatter([xt], net.propagate([xt]))
        #plt.scatter([x], [y], color='red')

       #print neuron.propagate(in_data)

#net.randomshutoff(0)
plt.plot(r, netfvec(r), color='red')
#plt.plot(r2, netfvec(r2), color='red')
#plt.plot(r2, target(r2), color='blue')
#plt.scatter([1],[1])
#plt.scatter([2],[2])
#plt.plot([1,2,3,4])
plt.show()
'''