from __future__ import print_function
from recurrentffnet import recurrentffnet
from ffnet import ffnet
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import random
from activation_functions import ActivationFunctions

import os
from neat import nn, population, statistics, activation_functions


def eval_fitness(genomes):
    for g in genomes:
        net = nn.create_feed_forward_phenotype(g)

        sum_square_error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):
            # Serial activation propagates the inputs through the entire network.
            output = net.serial_activate(inputs)
            sum_square_error += (output[0] - expected) ** 2

        # When the output matches expected for all inputs, fitness will reach
        # its maximum value of 1.0.
        g.fitness = 1 - sum_square_error


def target(x):
    return np.sin(x*8)*0.5+0.5

def netf(x):
    result=[]
    for xi in x:
        result.append(winner_net.serial_activate(xi))
    return result

# Network inputs and expected outputs.
xor_inputs = np.arange(0, 1, 0.05)
xor_outputs = target(xor_inputs)

xor_inputs=map(lambda x:[x,0],xor_inputs)

print(xor_inputs)
print(xor_outputs)

#xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
#xor_outputs = [0, 1, 1, 0]

print(xor_inputs)
print(xor_outputs)

def sinc(x):
    return 1.0 if x == 0 else np.sin(x) / x

# This sinc function will be available if my_sinc_function is included in the
# config file activation_functions option under the pheotype section.
# Note that sinc is not necessarily useful for this example, it was chosen
# arbitrarily just to demonstrate adding a custom activation function.
activation_functions.add('my_sinc_function', sinc)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'xor2_config')
pop = population.Population(config_path)
pop.run(eval_fitness, 300)

# Log statistics.
statistics.save_stats(pop.statistics)
statistics.save_species_count(pop.statistics)
statistics.save_species_fitness(pop.statistics)

print('Number of evaluations: {0}'.format(pop.total_evaluations))

# Show output of the most fit genome against training data.
winner = pop.statistics.best_genome()
print('\nBest genome:\n{!s}'.format(winner))
print('\nOutput:')
winner_net = nn.create_feed_forward_phenotype(winner)
for inputs, expected in zip(xor_inputs, xor_outputs):
    output = winner_net.serial_activate(inputs)
    print("expected {0:1.5f} got {1:1.5f}".format(expected, output[0]))




plt.plot(np.arange(0, 1, 0.05), netf(xor_inputs), color='red')
plt.plot(np.arange(0, 1, 0.05), target(np.arange(0, 1, 0.05)), color='red')

plt.show()