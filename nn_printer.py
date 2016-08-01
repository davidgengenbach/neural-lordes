from keras.utils.test_utils import layer_test

try:
    import matplotlib.pyplot as plt
except:
    raise

from activation_functions import ActivationFunctions

import networkx as nx


from ffnet import ffnet

def print_ffnet(nn):
    G = nx.Graph()
    pos = {}
    for layer_index, layer in enumerate(nn.layers):
        x = layer_index * 100
        for perceptron_index, perceptron in enumerate(layer):
            y = perceptron_index * 100
            name = '{},{}'.format(layer_index, perceptron_index)
            pos[name]=  [x, y]
            G.add_node(name)
            print name
            if layer_index != len(nn.layers) - 1:
                for next_perceptron_index, next_layer_perceptron in enumerate(nn.layers[layer_index + 1]):
                    next_name = '{},{}'.format(layer_index + 1, next_perceptron_index)
                    G.add_edge(name, next_name, weight=next_layer_perceptron.weights[perceptron_index])
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    #nx.draw_networkx_labels(G,pos)
    plt.show()

if __name__ == '__main__':
    args = [10, 10, 1, 3, ActivationFunctions.tanh]
    nn = ffnet(*args)
    print_ffnet(nn)
