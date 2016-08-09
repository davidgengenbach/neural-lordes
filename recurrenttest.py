from recurrentffnet import recurrentffnet
from ffnet import ffnet
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import random
from activation_functions import ActivationFunctions

args = [[1, 10,10, 1], ActivationFunctions.tanh]

net=recurrentffnet(*args)

#for i in range(200):
#    print net.propagate([1])
 #   net.learn([0.5],0.5)


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

for i in range(400):
    #print i
    alpha=0.05#random.uniform(0.000001, 0.01)
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

