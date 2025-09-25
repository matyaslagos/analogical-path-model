#!/usr/bin/env python3
import math
import custom_io as cio
from collections import defaultdict, Counter
from itertools import product

#-------------------------------------------------#
# Efficiency (normalized entropy) based weighting #
#-------------------------------------------------#

def node_efficiency(sequence_node):
    if not sequence_node.children.values():
        return 1
    total_freq = sequence_node.freq
    distr = []
    for child_node in sequence_node.children.values():
        distr.append(child_node.freq / total_freq)
    entropy = sum(prob * math.log(1 / prob, 2) for prob in distr)
    if entropy == 0:
        return 0
    max_entropy = math.log(len(sequence_node.children), 2)
    return entropy / max_entropy

def prob_neighbors_aux(sequence_node, parent_freq, prob_mass=1, path=[]):
    node_freq = sequence_node.freq
    if path:
        efficiency = node_efficiency(sequence_node)
        prob = efficiency * prob_mass * (node_freq / parent_freq)
        yield (path, prob)
    else:
        efficiency = 0
    for child_token, child_node in sequence_node.children.items():
        new_path = path + [child_token]
        child_prob_mass = (1 - efficiency) * prob_mass * (node_freq / parent_freq)
        yield from prob_neighbors_aux(child_node, node_freq, child_prob_mass, new_path)

def weighted_condprob(model, context, sequence):
    context_node = model.sequence_node(context)
    rem_prob_mass = 1
    curr_prob_mass = 0
    parent_freq = context_node.freq
    for token in sequence:
        child_node = context_node.children[token]
        child_freq = child_node.freq
        efficiency = min(max(0.5, node_efficiency(child_node)), 0.9)
        curr_prob_mass = rem_prob_mass * efficiency * (child_freq / parent_freq)
        rem_prob_mass = rem_prob_mass * (1 - efficiency) * (child_freq / parent_freq)
        parent_freq = child_freq
        context_node = child_node
    return curr_prob_mass