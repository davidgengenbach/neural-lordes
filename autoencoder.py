import numpy as np
import matplotlib.pyplot as plt
import json
from activation_functions import ActivationFunctions
from ffnet import ffnet

ALPHA = 0.05
INIT_AND_SAVE_WEIGHTS = False
LAYERS = [100, 100, 1, 50]
USED_ACTIVATION = ActivationFunctions.tanh

def deserialize(data):
    arguments = data['arguments']
    arguments.append(USED_ACTIVATION)
    nn = ffnet(*arguments)
    for index, layer in enumerate(data['layers']):
        for p_index, perceptron in enumerate(nn.layers[index]):
            perceptron.weights = layer[p_index]
    return nn

def serialize(nn):
    serialized_nn = {'arguments': nn.arguments}
    weights = []
    for layer in nn.layers:
        layer_weights = []
        for perceptron in layer:
            layer_weights.append(perceptron.weights)
        weights.append(layer_weights)
    serialized_nn['layers'] = weights
    return serialized_nn

def get_nn(INIT_AND_SAVE_WEIGHTS):
    if INIT_AND_SAVE_WEIGHTS:
        arguments = LAYERS
        arguments.append(USED_ACTIVATION)
        autoencoder = ffnet(*arguments)
        with open('nn.json', 'w+') as file:
            json.dump(serialize(autoencoder), file)
    else:
        with open('nn.json', 'r') as f:
            data = json.load(f)
        autoencoder = deserialize(data)
    return autoencoder

autoencoder = get_nn(INIT_AND_SAVE_WEIGHTS)

target_functions = [
    lambda x: np.sin(x*8)*0.9,
    lambda x: x*0.1,
    lambda x: np.tanh(x*8)*0.9,
    lambda x: np.tanh(-x*8)*0.9,
    lambda x: np.cos(x * 5) * 0.9
]

xrange=np.arange(0, 1, 0.01)
input = [target_function(xrange) for target_function in target_functions[0:4]]

def plotvec(res,c = 'green'):
    plt.plot(xrange, res, color=c)

for cycle in range(100):
    if cycle % 10 == 0:
            print cycle
    for inp in input:
        out=np.array(autoencoder.propagate(inp))

        plotvec(out,'green')

        autoencoder.learn(inp, ALPHA)
        for p in range(len(autoencoder.layers[1])):
            for w in range(len(autoencoder.layers[1][p].weights)):
                autoencoder.layers[1][p].weights[w]=autoencoder.layers[0][w].weights[p]

for inp in input:
    plotvec(inp,'blue')
    plotvec(autoencoder.propagate(inp), 'red')

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