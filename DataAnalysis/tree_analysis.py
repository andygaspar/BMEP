import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
from ast import literal_eval
import pydot
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes import Subplot
from matplotlib.colors import Normalize
from matplotlib.pyplot import subplots


class Root:
    def __init__(self):
        self.id = (0,)

        self.best = 10**5
        self.mean = 0
        self.var = 0

        self.visits = 0
        self.depth = 0
        self.is_expanded = False
        self.parent = None
        self.children = {}

    def add_tj(self, row):
        if row.obj_val < self.best:
            self.best = row.obj_val

        self.visits += 1

        tj = row.tj

        if tj[0] in self.children.keys():
            self.children[tj[0]].rollout(tj, row.obj_val)
        else:
            self.children[tj[0]] = Node(self, tj)

        if len(self.children) == 3:
            self.is_expanded = True


class Node:

    def __init__(self, parent,  tj_left):
        self.id = parent.id + (tj_left[0],)

        self.best = parent.best
        self.mean = parent.best
        self.var = 0
        self.visits = 1
        self.is_expanded = False

        self.depth = parent.depth + 1

        self.parent = parent
        tj_left = tj_left[1:]
        self.children = {tj_left[0]: Node(self, tj_left)} if tj_left else None

    def rollout(self, tj, obj_val):
        if obj_val < self.best:
            self.best = obj_val

        self.visits += 1

        self.mean = self.mean * (self.visits - 1) / self.visits + obj_val/self.visits
        self.var = self.var * (self.visits - 1) / self.visits + ((obj_val - self.mean)**2)/ (self.visits - 1)

        tj_left = tj[1:]
        if len(tj_left):
            if tj_left[0] in self.children.keys():
                self.children[tj_left[0]].rollout(tj_left, obj_val)
            else:
                self.children[tj_left[0]] = Node(self, tj_left)

        if self.children is not None and len(self.children) == (3 + self.depth * 2):
            self.is_expanded = True




class Tree:

    def __init__(self, df):
        self.root = Root()
        self.graph = nx.Graph()
        self.nodes = 0
        self.df = df
        self.min_val = 10**5
        self.max_val = -10**5
        self.c_map = cm.plasma
        self.norm = None



    def fill_tree(self):
        for i, row in self.df.iterrows():
            self.root.add_tj(row)
        self.root.mean = self.root.children[list(self.root.children.keys())[0]].mean
        self.root.var = self.root.children[list(self.root.children.keys())[0]].var

    def get_color(self, value):
        return self.c_map(self.norm(value))
    def add_node(self, node, value):
        val = node.best if value == 'best' else (node.mean if value == 'mean' else node.var)
        if val < self.min_val:
            self.min_val = val
        if val > self.max_val:
            self.max_val = val
        self.graph.add_node(node.id, color=val, size= 40*node.visits)
        self.nodes += 1
        if node.parent is not None:
            self.graph.add_edge(node.id, node.parent.id, color='black')
        if node.children is None:
            return
        for child in node.children.values():
            self.add_node(child, value)

    def make_graph(self, value='best', title='*', save=False):
        self.add_node(self.root, value)
        node = self.root
        while node.children is not None:
            node = min(list(node.children.values()), key=lambda t: t.best)
            self.graph.edges[(node.parent.id, node.id)]['color'] = 'red'
        self.norm = Normalize(vmin=self.min_val, vmax=self.max_val)
        self.show(title, value, save)

    def show(self, title, value, save):
        plt.rcParams["figure.figsize"] = (70, 50)
        plt.rcParams.update({'font.size': 40})
        tj = self.df[self.df.obj_val == self.df.obj_val.min()].trajectory.iloc[0]

        fig, ax = subplots()
        g = nx.convert_node_labels_to_integers(self.graph, label_attribute='node_label')
        pos = nx.drawing.nx_pydot.graphviz_layout(g, prog="dot")
        node_colors = [self.get_color(g.nodes[node]["color"]) for node in g.nodes]
        edge_colors = [g.edges[edge]["color"] for edge in g.edges]
        sizes = [g.nodes[node]["size"] for node in g.nodes]
        nx.draw(g, ax=ax, pos=pos, node_size=sizes, node_color=node_colors, edge_color=edge_colors)
        plt.title('Iteration ' + str(title) + ' ' + value + '\n' + tj)
        if save:
            plt.savefig('DataAnalysis/plots/' + value + '_' + title, bbox_inches='tight')
        else:
            plt.show()


    def add_expanded_node(self, node, value):
        val = node.best if value == 'best' else (node.mean if value == 'mean' else node.var)
        if val < self.min_val:
            self.min_val = val
        if val > self.max_val:
            self.max_val = val
        self.graph.add_node(node.id, color=val, size=100)
        self.nodes += 1
        if node.parent is not None:
            self.graph.add_edge(node.id, node.parent.id, color='black')
        if not node.is_expanded:
            return
        for child in node.children.values():
            self.add_expanded_node(child, value)

    def make_expanded(self, tile, value='best', save=False):
        self.add_expanded_node(self.root, value)
        self.norm = Normalize(vmin=self.min_val, vmax=self.max_val)
        node = self.root
        while node.is_expanded:
            node = min(list(node.children.values()), key=lambda t: t.best)
            self.graph.edges[(node.parent.id, node.id)]['color'] = 'red'
        self.show(tile, value, save)


df = pd.read_csv('DataAnalysis/40_10.csv')
df['tj'] = df.trajectory.apply(lambda t: literal_eval(t))
df.rename(columns={"is_best": 'single_run_best'}, inplace=True)

df_ = df[df.iteration == 0]

tree = Tree(df)
tree.fill_tree()
tree.make_expanded('exapanded_all_40', 'best', save=True)

df = df[df.single_run_best == True]

#
# df_ = df[df.single_run_best == True]
# for val in ["best", "var"]:
#     for i in range(10):
#         dff = df_[df_.iteration==i]
#         tree = Tree(dff)
#         tree.fill_tree()
#         tree.make_graph(value=val, title='40_' + str(i), save=True)







