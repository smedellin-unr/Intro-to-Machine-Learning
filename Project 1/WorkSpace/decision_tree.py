import numpy as np
from math import log
from scipy import spatial
from random import randint
from enum import Enum
from abc import ABCMeta, abstractmethod

# TODO: how to read in a text file in python
# Have it output features and labels in correct numpy format
def read_features_labels(filepath:str=None) -> tuple:

    with open(filepath) as f:
        lines = f.readlines()

    # separate features and labels
    features_from_text = lines[0].rstrip('\n').split(';')
    labels_from_text = lines[1].rstrip('\n')

    # make numpy 2D array of features
    n_features = len(features_from_text[0].split(','))
    X = np.empty((0, n_features), bool)

    for feature_set in features_from_text:
        X = np.append(X, np.array([np.fromstring(feature_set,dtype=bool,sep=',')]), axis=0)

    # make numpy 1D array of labels
    Y = np.array(np.fromstring(labels_from_text,dtype=bool,sep=';'))

    return (X,Y)

class Leaf:
    def __init__(
        self,
        decision:bool,
        side: bool,
    ):
        self.decision = decision
        self.side = side

    def __str__(self):
        if self.side:
            return 'Right Leaf: Yes' if self.decision else 'Right Leaf: No'
        else:
            return 'Left Leaf: Yes' if self.decision else 'Left Leaf: No'

class Node:
    def __init__(
        self,
        X: np.ndarray=None, 
        Y: np.array=None, 
        max_depth: int=None,
        depth: int=None,
        node_type: str=None,
        leaf_decision: bool = None,
        leaf_side: bool = None,
    ) :
        self.Y = Y if Y is not None else np.zeros(shape=(2,4))# 1D array
        self.X = X if X is not None else np.zeros(shape=(2,4))# 2D array

        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]

        self.xy_matrix = np.column_stack((X,Y)) # features and labels concatenated into one 2D matrix

        self.node_type = node_type if node_type else 'root_node'
        self.max_depth = max_depth if max_depth else self.n_features
        self.depth = depth if depth else 0

        # TODO: Entropy at the node level
        self.entropy = Node.get_entropy(self.Y)

        # Can be decision node of leaf
        self.left = None 
        self.right = None

        self.best_feature = 0
        self.best_information_gain = 0.0

        self.leaf_side = leaf_side if leaf_side else None
        self.leaf_decision = leaf_decision if leaf_decision else None

    @staticmethod
    def get_entropy(Y:np.array) -> np.float16:
        pc = np.sum(Y) / Y.size
        if pc == 0.0 or pc == 1.0:
            return 0
        return -(1.0 - pc) * log((1.0 - pc), 2) - (pc) * log((pc), 2)

    @staticmethod
    def get_IG():
        pass

    @staticmethod
    def get_partitions(X: np.ndarray, Y:np.array) -> tuple:
        '''
        See which rows are labeled with true and false
        and split them accordingly
        '''
        true_rows = np.empty((0, X.shape[1]), bool)
        false_rows = np.empty((0, X.shape[1]), bool)
        for rows,labels in zip(X,Y):
            if labels == True:
                true_rows = np.vstack((true_rows, rows))
            else:
                false_rows = np.vstack((false_rows, rows))
        return (true_rows, false_rows)

    @staticmethod
    def get_split_feature(X: np.ndarray, Y: np.array, question: bool=True) -> np.array :
        '''
        extract label data from splits
        '''
        label_array = np.empty((0, X.shape[0]), bool)
        for row, label in zip(X, Y):
            for column in range(X.shape[1]):
                if row[column] == question:
                    np.append(label_array, True)
                else:
                    np.append(label_array, False)
        return label_array

    def best_split(self) -> tuple:
        '''
        Determine the feature yields
        the most information gain
        '''
        best_information_gain = 0.0
        best_feature = None
        # leaf decision assignment
        best_right_leaf_estimate = None
        best_left_leaf_estimate = None
        leaf_side = None
        leaf_decision = None
        # used for assessing where split will occur
        best_right_entropy = 0.0
        best_left_entropy = 0.0

        print(self.xy_matrix)
        print('\n')
        for column in range(self.n_features):
            left_entropy = right_entropy = left_feature_size = right_feature_size = total_size = 0
            right_leaf_estimate = left_leaf_estimate = None
            # calculate entropy for left branch
            left_label = self.xy_matrix[self.xy_matrix[:,column] == False][:, -1]
            if left_label.size == 0:
                # No 'NO' labels
                pass
            else:
                left_feature_size = left_label.shape[0]
                left_entropy = Node.get_entropy(left_label)
                left_leaf_estimate = True if np.sum(left_label) / left_label.shape[0] > 0.5 else False
                #print("left entropy: ",left_entropy)
                #print("shape: ",temp_label.shape[0])
                #print(temp_label)
                #print("the estimate: ",left_leaf_estimate)

            # calculate entropy for right branch
            right_label = self.xy_matrix[self.xy_matrix[:,column] == True][:, -1]
            if right_label.size == 0:
                # No 'YES' labels
                pass
            else:
                right_feature_size = right_label.shape[0]
                right_entropy = Node.get_entropy(right_label)
                right_leaf_estimate = True if np.sum(right_label) / right_label.shape[0] > 0.5 else False
                #print("right entropy: ",right_entropy)

            # calculate Information Gain for split
            total_size = left_feature_size + right_feature_size
            information_gain = self.entropy - ((left_feature_size / total_size) * left_entropy + (right_feature_size / total_size) * right_entropy)
            '''
            print("feature {} and IG {}".format(column + 1, information_gain))
            print("Left Entropy: ", left_entropy)
            print("left proportion: ", left_feature_size / total_size)
            print("Right Entropy: ", right_entropy)
            print("right proportion: ", right_feature_size / total_size)
            print("Information gain: ",information_gain)
            '''

            if information_gain >= best_information_gain:
                best_feature, best_information_gain = column, information_gain
                best_right_leaf_estimate = right_leaf_estimate
                best_right_entropy = right_entropy
                best_left_leaf_estimate = left_leaf_estimate
                best_left_entropy = left_entropy
                
        if best_right_entropy >= best_left_entropy:
            leaf_side = True
            leaf_decision = best_right_leaf_estimate
        else:
            leaf_side = False                   # TODO: make this prettier
            leaf_decision = best_left_leaf_estimate

        return (best_feature, best_information_gain, leaf_side, leaf_decision)

    def leaf_assignment(self,leaf_side: bool, leaf_assignment: bool):
        pass

    def grow_tree(self):
        '''
        Recursive function
        to build decision tree
        '''

        best_feature, best_information_gain, leaf_side, leaf_decision = self.best_split()

        if (best_feature is not None) and (best_information_gain != 0):
            self.best_feature = best_feature
            self.best_information_gain = best_information_gain
            print("best feature:", best_feature)

            # if leaf is on the right decision branch...
            if not leaf_side:

                # make the left decision branch a node
                left_side_dataset = np.copy(self.xy_matrix[self.xy_matrix[:,best_feature] == False])
                left = Node(
                    X=left_side_dataset[:,:-1],
                    Y=left_side_dataset[:,-1],
                    depth=self.depth+1,
                    max_depth=self.max_depth,
                    node_type='left_node'
                )
                
                self.left = left
                print(self.left.node_type)

                self.left.grow_tree()
                
            # if leaf is on the left decision branch
            else:
                right_side_dataset = np.copy(self.xy_matrix[self.xy_matrix[:,best_feature] == True])
                right = Node(
                    X=right_side_dataset[:,:-1],
                    Y=right_side_dataset[:,-1],
                    depth=self.depth+1,
                    max_depth=self.max_depth,
                    node_type='right_node'
                )

                self.right = right
                print(self.right.node_type)
                self.right.grow_tree()
        
        if self.right is None : 
            self.right = Node(
                node_type='leaf', 
                leaf_decision=leaf_decision, 
                leaf_side=leaf_side
            )

        if self.left is None : 
            self.left = Node(
                node_type='leaf', 
                leaf_decision=leaf_decision, 
                leaf_side=leaf_side
            )

        
def print_tree(node: Node):
    if (node.leaf_decision is not None) and (node.leaf_side is not None):
        response = 'Yes' if node.leaf_decision else 'No'
        side = 'Right' if node.leaf_side else 'Left'
        print('{} Leaf: {}'.format(side, response))
        return
    if 'node' in node.node_type:
        print(node.node_type)
        print_tree(node.left)
        print_tree(node.right)
    


if __name__ == "__main__":

    X, Y = read_features_labels('data_set_TV.txt')
    n_=Node(X=X, Y=Y)
    print_tree(n_)
    '''
    best_feature = best_information_gain = 0
    print('root node')
    n_.grow_tree()
    print_tree(n_)
    #print(best_feature, best_information_gain)
    #print(n_.best_feature, n_.best_information_gain)
    #print(best_feature, best_information_gain)
    t, f = Node.get_partitions(X,Y)
    '''