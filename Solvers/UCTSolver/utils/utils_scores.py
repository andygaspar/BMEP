import numpy as np


def average_score_normalised(node):
    normalisation = max([child.average().item() for child in node._children])
    shift = min([child.average().item() for child in node._children])
    scores = [
        1 - (child.average().item() - shift) / normalisation + node._c * np.sqrt(np.log(node._n_visits) / child.n_visits())
        for child in node._children]
    return scores


def max_score_normalised(node):
    normalisation = max([child._val.item() for child in node._children])
    shift = min([child._val.item() for child in node._children])
    scores = [
        1 - (child._val.item() - shift) / normalisation + node._c * np.sqrt(np.log(node._n_visits) / child.n_visits())
        for child in node._children]
    return scores

def average_score_normalised_dict(node):
    normalisation = max([node.children[child].average_obj_val.item() for child in node.children.keys()])
    shift = min([node.children[child].average_obj_val.item() for child in node.children.keys()])
    scores = [
        1 - (node.children[child].average_obj_val.item() - shift) / normalisation +
        node.c * np.sqrt(np.log(node.n_visits) / node.children[child].n_visits)
        for child in node.children.keys()]
    return scores


def max_score_normalised_dict(node):
    normalisation = max([node.children[child].val.item() for child in node.children.keys()])
    shift = min([node.children[child].val.item() for child in node.children.keys()])
    scores = [
        1 - (node.children[child].val.item() - shift) / normalisation +
        node.c * np.sqrt(np.log(node.n_visits) / node.children[child].n_visits)
        for child in node.children.keys()]
    return scores
