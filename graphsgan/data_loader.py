import sys

import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp



def load_data(dataset_str = 'cora'):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # normalize
    features = normalize(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # network_emb = pros(adj)
    # network_emb = 0
    

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]     # onehot
    
    idx_train = range(len(y))       # training data index
    idx_val = range(len(y), len(y)+500)     # validation data index
    idx_test = test_idx_range.tolist()      # test data index 

    features = np.array(features.todense())
    return adj, features, labels, idx_train, idx_val, idx_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test = load_data('cora')
    print('adj shape', adj.shape)        # adjacency matrix with Shape(2708, 2708)
    print('feature shape', features.shape)      # feature matrix, Shape(2708, 1433)
    print('label shape', labels.shape)      #label matrix, Shape(2708, 7)
    print('train/validation/test split: %s/%s/%s'%(len(idx_train), len(idx_val), len(idx_test)) )
