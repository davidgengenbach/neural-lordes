import numpy as np
import random

class Perceptron:
    def __init__(self, inpsize,id):
        self.weights = [random.uniform(-3, 3) for x in range(inpsize)]
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
        self.output=self.activationfunction(np.dot(inp, self.weights)+self.bias)
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

