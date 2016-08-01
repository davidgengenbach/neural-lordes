try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx
from ffnet import ffnet
from activation_functions import ActivationFunctions

def draw_ffnet(nn):
    (nodes, edges, pos) = get_graph_data(nn)
    draw_graph(edges, pos)

def get_graph_data(nn):
    edges = []
    nodes = []
    pos = {}
    for layer_index, layer in enumerate(nn.layers):
        x = layer_index
        layer_len = len(layer)
        for perceptron_index, perceptron in enumerate(layer):
            y = perceptron_index
            name = '{},{}'.format(layer_index, perceptron_index)
            nodes.append(name)
            pos[name]= [x, y / float(layer_len - 1)]
            if layer_index != len(nn.layers) - 1:
                for next_perceptron_index, next_layer_perceptron in enumerate(nn.layers[layer_index + 1]):
                    next_name = '{},{}'.format(layer_index + 1, next_perceptron_index)
                    weight = next_layer_perceptron.weights[perceptron_index]
                    edges.append({'from': name, 'to': next_name, 'weight': weight})
    return (nodes, edges, pos)

# @see http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
# @see http://matplotlib.org/users/colormaps.html
def draw_graph(edges, pos, color_map = plt.cm.Blues):
    DEBUG = False
    G = nx.Graph()
    key_fn = lambda x: x['weight']
    min_weight = min(edges, key=key_fn)['weight']
    max_weight = max(edges, key=key_fn)['weight']
    for edge in edges:
        G.add_edge(edge['from'], edge['to'], weight = edge['weight'])
        # Normalize
        edge['normalized_weight'] = (edge['weight'] - min_weight) / (max_weight - min_weight)
    colors = [int(len(edges) * x['normalized_weight']) for x in edges]
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw(G,pos,node_color='#A0CBE2', edge_color=colors, weight=5, edge_cmap=color_map,with_labels=False)
    if DEBUG:
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.show()

if __name__ == '__main__':
    args = [10, 10, 2, 3, ActivationFunctions.tanh]
    nn = ffnet(*args)
    draw_ffnet(nn)
