# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2003-2007 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://framagit.org/gabylaunay/IMTreatment

# This file is part of IMTreatment.

# IMTreatment is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# IMTreatment is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt


def FDG_optimization(nodes, edges, edge_weights, node_values):
    try:
        import networkx as nx
    except ImportError:
        raise Exception("You need to install `networkx` to use this"
                        " functionality.")
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for edge, weight in zip(edges, edge_weights):
        graph.add_edge(*edge, weight=weight)
    # set initial position
    init_pos = {node: (node_values[node], np.random.rand(1)) for node in nodes}
    node_pos = nx.fruchterman_reingold_layout(graph, dim=2, weight="weight",
                                              iterations=500, pos=init_pos)
    plt.figure()
    nx.draw(graph, with_labels=True, node_color=node_values, pos=init_pos)
    plt.figure()
    nx.draw(graph, with_labels=True, node_color=node_values, pos=node_pos)
    plt.figure()
    nx.draw_spring(graph)
    return node_pos
