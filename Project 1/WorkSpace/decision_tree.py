import numpy as np
from math import log
from scipy import spatial
from random import randint

# TODO: how to read in a text file in python
# Have it output features and labels in correct numpy format
def read_features_labels(filepath:str=None) -> tuple:

    with open(filepath) as f:
        lines = f.readlines()

    # separate features and labels
    features_from_text = lines[0].rstrip('\n').split(';')
    labels_from_text = lines[1].rstrip('\n')

    # make numpy 2D array of features
    number_of_features = len(features_from_text[0].split(','))
    X = np.empty((0, number_of_features), bool)

    for feature_set in features_from_text:
        X = np.append(X, np.array([np.fromstring(feature_set,dtype=bool,sep=',')]), axis=0)

    # make numpy 1D array of labels
    Y = np.array(np.fromstring(labels_from_text,dtype=bool,sep=';'))

    return (X,Y)


class Node:
    def __init__(
        self,
        Y: np.array, 
        X: np.ndarray, 
        max_depth=None,
        depth=None,
    ) :
        self.Y = Y
        self.X = X

        self.max_depth = max_depth if max_depth else 3
        self.depth = depth if depth else 0

        # TODO: Entropy at the node level
        self.entropy = Node.get_entropy(self.Y)

        self.left = None
        self.right = None

        self.best_feature = None

    @staticmethod
    def get_entropy(Y:np.array=None) -> np.float16:

        pc = np.sum(Y) / len(Y)
        pc_other = 1.0 - pc

        return -pc * log(pc, 2) - pc_other * log(pc_other, 2)

    @staticmethod
    def get_IG():
        pass

    @staticmethod
    def get_partitions(X: np.ndarray, Y:np.array) -> tuple:
        '''
        See which rows correspond with true and false
        and split them
        '''
        true_rows = np.empty((0, 4), bool)
        false_rows = np.empty((0, 4), bool)
        for rows,labels in zip(X,Y):
            if labels == True:
                true_rows = np.vstack((true_rows, rows))
            else:
                false_rows = np.vstack((false_rows, rows))
        return (true_rows, false_rows)


if __name__ == "__main__":

    X, Y = read_features_labels('data_1.txt')
    t, f = Node.get_partitions(X,Y)
    print(t)
    print(f)
    print(Node.get_entropy(Y))
    print(X)
    print(Y)