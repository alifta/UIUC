# Imports
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from .io import *
from .utils import path_edit


# BFS
def simple_bfs(graph, source):
    """A fast BFS node generator"""
    adj = graph.adj
    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                seen.add(v)
                nextlevel.update(adj[v])
    return seen


class SampleGraph():
    def __init__(self) -> None:
        self.graph = None

    def load_data(self, download=False):
        URL = [
            'https://gist.githubusercontent.com/brooksandrew/f989e10af17fb4c85b11409fea47895b/raw/a3a8da0fa5b094f1ca9d82e1642b384889ae16e8/nodelist_sleeping_giant.csv',
            'https://gist.githubusercontent.com/brooksandrew/e570c38bcc72a8d102422f2af836513b/raw/89c76b2563dbc0e88384719a35cba0dfc04cd522/edgelist_sleeping_giant.csv'
        ]
        FILE = path_edit(
            folder_name=DATA,
            folder_label='sleeping_giant',
            file_names=['nodelist.csv', 'edgelist.csv']
        )
        if download:
            nodelist = pd.read_csv(URL[0])
            edgelist = pd.read_csv(URL[1])
            nodelist.to_csv(FILE[0])
            edgelist.to_csv(FILE[1])
        else:
            nodelist = pd.read_csv(FILE[0])
            edgelist = pd.read_csv(FILE[1])
        return nodelist, edgelist

    def create_sleeping_giant(self, stat=True, save=True):
        nodelist, edgelist = self.load_data()
        # Empty graph
        g = nx.Graph()
        g.name = 'Sleeping Giant Trail Map Graph'
        for i, elrow in edgelist.iterrows():
            # Add edge
            g.add_edge(elrow[0], elrow[1])
            # Update edge attributes
            g[elrow[0]][elrow[1]].update(elrow[2:].to_dict())
        # Update node attributes
        for i, nlrow in nodelist.iterrows():
            g.nodes[nlrow['id']].update(nlrow[1:].to_dict())
        if stat:
            print(nx.info(g))
        if save:
            FILE = path_edit(
                folder_name=DATA,
                folder_label='sleeping_giant',
                file_names=['sleeping_giant.gpickle']
            )
            # Undirected
            # nx.write_gpickle(g, FILE[0])
            # Directed
            nx.write_gpickle(g.to_directed(), FILE[0])
        self.graph = g
        return g

    def visualize_sleeping_giant(self):
        g = self.graph
        # Node positions from node features
        # X and Y info are from dataset
        node_positions = {
            node[0]: (node[1]['X'], -node[1]['Y'])
            for node in g.nodes(data=True)
        }
        # Node labels / names
        c = 0
        node_labels = {}
        for node in g.nodes():
            node_labels[node] = c
            # Save index in node for future reference
            g.nodes[node]['idx'] = c
            c += 1
        # Edge colors
        edge_colors = [e[2]['color'] for e in g.edges(data=True)]
        # Plot the graph
        plt.figure(figsize=(16, 12), dpi=400)
        nx.draw(
            g,
            pos=node_positions,
            labels=node_labels,
            # node_size=400,
            # node_color='black',
            font_color='white',
            edge_color=edge_colors
        )
        plt.title(g.name, size=16)
        plt.show()