import random
from gensim.models import Word2Vec
import networkx as nx
import numpy as np

def walker(G, walk_length, start_node):
    walk = [str(start_node)]

    while len(walk) < walk_length:
        cur = int(walk[-1])
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(str(random.choice(cur_nbrs)))
        else:
            break
    return walk
# From https://github.com/shenweichen/GraphEmbedding/blob/master/ge/walker.py
def _simulate_walks(G, nodes, num_walks, walk_length):
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for v in nodes:
            walks.append(walker(G, walk_length=walk_length, start_node=v))
    return walks

def walk2vec(G, num_walks, walk_length):
    walks = _simulate_walks(G, list(G.nodes()), num_walks, walk_length)
    return Word2Vec(walks,sg=1,hs=1)



def neighbour_fushion(node, nblist, fmatrix, alpha):
    sum = np.zeros(len(fmatrix[0]))
    for idx in nblist:
        sum = sum + fmatrix[idx]
    return sum * (1 - alpha) / len(nblist) + alpha * fmatrix[node]