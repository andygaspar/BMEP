import numpy as np


def average_score_normalised(node):
    normalisation = max([child.average() for child in node._children])
    shift = min([child.average() for child in node._children])
    scores = [
        1 - (child.average() - shift) / normalisation + node._c * np.sqrt(np.log(node._n_visits) / child.n_visits())
        for child in node._children]
    return scores


def max_score_normalised(node):
    normalisation = max([child._val for child in node._children])
    shift = min([child._val for child in node._children])
    scores = [
        1 - (child._val - shift) / normalisation + node._c * np.sqrt(np.log(node._n_visits) / child.n_visits())
        for child in node._children]
    return scores
