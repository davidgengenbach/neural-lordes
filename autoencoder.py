import json
import sys
import pylab
import ffnet_utils
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from activation_functions import ActivationFunctions
from ffnet import ffnet

LEARN_RATE = 0.05
NN_FILE = 'nn.json'

# TODO: ...
factor = 0.02
xrange = np.arange(0, 1, factor)
INPUT_LAYER_SIZE = int(1 / factor)
CYCLE_COUNT = 100

if INPUT_LAYER_SIZE != len(xrange):
    print "Not a valid factor - (1 / factor) should be a whole number"
    sys.exit(1)

INIT_AND_SAVE_WEIGHTS = True
# Parameters for the ffnet
# Only used when INIT_AND_SAVE_WEIGHTS is True
# array of layer sizes
NN_ARGUMENTS = [[INPUT_LAYER_SIZE, 30, INPUT_LAYER_SIZE]]
USED_ACTIVATION = ActivationFunctions.tanh


def get_nn(init_and_save_weights):
    if init_and_save_weights:
        arguments = NN_ARGUMENTS
        arguments.append(USED_ACTIVATION)
        autoencoder = ffnet(*arguments)
        with open(NN_FILE, 'w+') as file:
            json.dump(ffnet_utils.serialize(autoencoder), file)
    else:
        with open(NN_FILE, 'r') as file:
            data = json.load(file)
        autoencoder = ffnet_utils.deserialize(data, USED_ACTIVATION)
    return autoencoder

autoencoder = get_nn(INIT_AND_SAVE_WEIGHTS)

target_functions = [
    lambda x: np.sin(x*8)*0.9,
    lambda x: x*0.1,
    lambda x: np.tanh(x*8)*0.9,
    lambda x: np.tanh(-x*8)*0.9,
    lambda x: np.cos(x * 5) * 0.9
]

input = [target_function(xrange) for target_function in target_functions]


def plotvec(res, c='green'):
    plt.plot(xrange, res, color=c)


def clear_plot():
    plt.clf()
    plt.cla()

pylab.show()
gs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[1, 5, 5])
subplot_top = plt.subplot(gs[0])
subplot_middle = plt.subplot(gs[1])
subplot_bottom = plt.subplot(gs[2])


for cycle in range(CYCLE_COUNT):
    for index, inp in enumerate(input):
        out = np.array(autoencoder.propagate(inp))

        autoencoder.learn(inp, LEARN_RATE)

        # Copy weights from second layer to first layer
        for p, second_layer_perceptron in enumerate(autoencoder.layers[1]):
            for w, weight in enumerate(second_layer_perceptron.weights):
                second_layer_perceptron.weights[w] = autoencoder.layers[0][w].weights[p]

        clear_plot()
        plt.subplot(subplot_top)
        plt.axis('off')
        plt.text(0, 0, 'CurrentIteration: {}'.format(cycle * len(input) + index))

        plt.subplot(subplot_middle)
        plotvec(out, 'green')
        for inp in input:
            plotvec(inp, 'blue')
            plotvec(autoencoder.propagate(inp), 'red')

        plt.subplot(subplot_bottom)
        ffnet_utils.draw_ffnet(autoencoder)

        pylab.draw()
        plt.pause(0.01)


# If you're patient enough...
for inp in input:
    plotvec(inp, 'blue')
    plotvec(autoencoder.propagate(inp), 'red')

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
