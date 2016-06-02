#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
IMTreatment module

    Auteur : Gaby Launay
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pdb


def FDG_optimization(nodes, edges, edge_weights, node_values):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for edge, weight in zip(edges, edge_weights):
        graph.add_edge(*edge, weight=weight)
    # set initial position
    init_pos = {node: (nodes_values[node], np.random.rand(1)) for node in nodes}
    # pdb.set_trace()
    node_pos = nx.fruchterman_reingold_layout(graph, dim=2, weight="weight",
                                              iterations=500, pos=init_pos)
    plt.figure()
    nx.draw(graph, with_labels=True, node_color=node_values, pos=init_pos)
    plt.figure()
    nx.draw(graph, with_labels=True, node_color=node_values, pos=node_pos)
    plt.figure()
    nx.draw_spring(graph)
    return node_pos





if __name__ == '__main__':
    nmb_nodes = 100
    nodes = range(nmb_nodes)
    nodes_values = np.array([2]*40 + [6]*60) + np.random.rand(nmb_nodes)
    weights = []
    links = []
    for i in range(nmb_nodes):
        for j in range(nmb_nodes):
            if i >= j:
                continue
            weights.append(abs(nodes_values[i] - nodes_values[j]))
            links.append([nodes[i], nodes[j]])



    weights = 1 + np.max(weights) - np.array(weights)
    pos = FDG_optimization(nodes, links, weights, nodes_values)
    plt.show()
    
    # plt.figure()
    # for node in pos.keys():
    #     plt.plot(*pos[node], marker='o', color='w')
    #     plt.text(*pos[node], s="{}".format(node))
    # for link in links:
    #     pos1 = pos[link[0]]
    #     pos2 = pos[link[1]]
    #     plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color='k')
    # plt.show()
        
