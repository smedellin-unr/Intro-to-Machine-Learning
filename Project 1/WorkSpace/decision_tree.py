import numpy as np
import math
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
        Y: list, 
        X: np.ndarray, 
        max_depth=None,
        depth=None,
    ) :
        self.Y = Y
        self.X = X

        self.max_depth = max_depth if max_depth else 3
        self.depth = depth if depth else 0

        # TODO: Entropy at the node level

        self.left = None
        self.right = None

        self.best_feature = None

    @staticmethod
    def get_entropy(Y:np.array=None) -> np.float16:

        pc = np.sum(Y) / len(Y)
        neg_pc = 1.0 - pc

        



if __name__ == "__main__":

    X, Y = read_features_labels('data_1.txt')

    print(Node.get_entropy(Y))
    print(X)
    print(Y)