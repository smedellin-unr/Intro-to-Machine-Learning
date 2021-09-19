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
    n_features = len(features_from_text[0].split(','))
    X = np.empty((0, n_features), bool)

    for feature_set in features_from_text:
        X = np.append(X, np.array([np.fromstring(feature_set,dtype=bool,sep=',')]), axis=0)

    # make numpy 1D array of labels
    Y = np.array(np.fromstring(labels_from_text,dtype=bool,sep=';'))

    return (X,Y)


class Node:
    def __init__(
        self,
        X: np.ndarray, 
        Y: np.array, 
        max_depth=None,
        depth=None,
    ) :
        self.Y = Y # 1D array
        self.X = X # 2D array

        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]

        self.max_depth = max_depth if max_depth else self.n_features # TODO: make sure this is legit
        self.depth = depth if depth else 0

        # TODO: Entropy at the node level
        self.entropy = Node.get_entropy(self.Y)

        self.left = None
        self.right = None

        self.best_feature = None

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
        #true_rows, false_rows = Node.get_partitions(self.X, self.Y)
        xy_matrix = np.column_stack((self.X,self.Y))
        best_feature = None
        best_value = None
        best_information_gain = 0.0

        for column in range(self.n_features):
            left_entropy = right_entropy = left_feature_size = right_feature_size = total_size = 0
            # calculate entropy for left branch
            temp_label = xy_matrix[xy_matrix[:,column] == False][:, -1]
            if temp_label.size == 0:
                # No 'NO' labels
                pass
            else:
                left_feature_size = temp_label.shape[0]
                left_entropy = Node.get_entropy(temp_label)

            # calculate entropy for right branch
            temp_label = xy_matrix[xy_matrix[:,column] == True][:, -1]
            if temp_label.size == 0:
                # No 'YES' labels
                pass
            else:
                right_feature_size = temp_label.shape[0]
                right_entropy = Node.get_entropy(temp_label)

            # calculate Information Gain for split
            total_size = left_feature_size + right_feature_size
            information_gain = self.entropy - ((left_feature_size / total_size) * left_entropy + (right_feature_size / total_size) * right_entropy)

            if information_gain >= best_information_gain:
                best_feature, best_information_gain = column, information_gain
                
        return (best_feature, best_information_gain)
        

if __name__ == "__main__":

    X, Y = read_features_labels('data_1.txt')
    n_=Node(X=X, Y=Y)
    best_feature = best_information_gain = 0
    best_feature, best_information_gain = n_.best_split()
    print(best_feature, best_information_gain)
    t, f = Node.get_partitions(X,Y)