try:
    import matplotlib.pyplot as plt
except:
    raise

import pylab
import networkx as nx
from ffnet import ffnet
from activation_functions import ActivationFunctions


def draw_ffnet(nn):
    '''You have to execute plt.show() by yourself. this just draws'''
    (nodes, edges, pos) = get_graph_data(nn)
    draw_graph(edges, pos)


def get_graph_data(nn):
    edges = []
    nodes = []
    pos = {}

    input_layer = range(nn.input_size)
    layers = [input_layer] + nn.layers
    for layer_index, layer in enumerate(layers):
        x = layer_index
        layer_len = len(layer)
        for perceptron_index, perceptron in enumerate(layer):
            y = perceptron_index
            name = '{},{}'.format(layer_index, perceptron_index)
            nodes.append(name)
            pos[name] = [x, y / float(layer_len - 1)]
            if layer_index != len(layers) - 1:
                for next_perceptron_index, next_layer_perceptron in enumerate(layers[layer_index + 1]):
                    next_name = '{},{}'.format(layer_index + 1, next_perceptron_index)
                    weight = next_layer_perceptron.weights[perceptron_index]
                    edges.append({'from': name, 'to': next_name, 'weight': weight})
    return (nodes, edges, pos)


def draw_graph(edges, pos, color_map=plt.cm.Blues, node_color='#A0CBE2'):
    # @see http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    # @see http://matplotlib.org/users/colormaps.html
    DEBUG = False
    G = nx.Graph()
    key_fn = lambda x: x['weight']
    min_weight = min(edges, key=key_fn)['weight']
    max_weight = max(edges, key=key_fn)['weight']
    for edge in edges:
        # Normalize the weight so it's between [0, 1]
        edge['normalized_weight'] = (edge['weight'] - min_weight) / (max_weight - min_weight)
        G.add_edge(edge['from'], edge['to'], weight=edge['weight'], normalized_weight=edge['normalized_weight'])
    # assigns a color relative to the weight
    colors = [int(len(edges) * x['normalized_weight']) for x in edges]
    if DEBUG:
        labels = nx.get_edge_attributes(G, 'normalized_weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw(G, pos, node_color=node_color, node_size=6, edge_color=colors,
            weight=3, edge_cmap=color_map, with_labels=False)


def deserialize(data, used_activation = None):
    arguments = data['arguments']
    if used_activation is None:
        used_activation = getattr(ActivationFunctions, data['activation_function'])
    arguments.append(used_activation)
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
    serialized_nn['activation_function'] = nn.activation_function.__name__
    return serialized_nn

if __name__ == '__main__':
    args = [[10, 3, 10], ActivationFunctions.tanh]
    nn = ffnet(*args)
    draw_ffnet(nn)
    plt.show()
