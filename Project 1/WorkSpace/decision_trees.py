import numpy as np
from math import log


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
        X: np.ndarray=None, 
        Y: np.array=None, 
        max_depth: int=None,
        depth: int=0,
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
        self.max_depth = self.n_features if max_depth == -1 else max_depth
        self.depth = depth

        self.entropy = Node.get_entropy(self.Y)

        # Can be decision node of leaf
        self.left = None 
        self.right = None

        self.best_feature = 0
        self.best_information_gain = 0.0

        self.leaf_side = leaf_side
        self.leaf_decision = leaf_decision

    @staticmethod
    def get_entropy(Y:np.array) -> np.float16:
        pc = np.sum(Y) / Y.size
        if pc == 0.0 or pc == 1.0:
            return 0
        return -(1.0 - pc) * log((1.0 - pc), 2) - (pc) * log((pc), 2)

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
        # terminal node detection
        terminal_node = False

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
            
            if information_gain >= best_information_gain:
                best_feature, best_information_gain = column, information_gain
                best_right_leaf_estimate = right_leaf_estimate
                best_right_entropy = right_entropy
                best_left_leaf_estimate = left_leaf_estimate
                best_left_entropy = left_entropy
        if best_right_entropy >= best_left_entropy:
            leaf_side = False
            leaf_decision = best_left_leaf_estimate
        else:
            leaf_side = True
            leaf_decision = best_right_leaf_estimate

        if (best_right_entropy == 0.0 and best_left_entropy == 0.0) or (self.max_depth == self.depth):
            terminal_node = True

        return (best_feature, best_information_gain, leaf_side, leaf_decision, terminal_node)

    def leaf_assignment(self,leaf_side: bool, leaf_assignment: bool):
        pass

    def grow_tree(self):
        '''
        Recursive function
        to build decision tree
        '''
        best_feature, best_information_gain, leaf_side, leaf_decision, terminal_node = self.best_split()

        if (best_feature is not None) and (best_information_gain != 0): #and (self.depth < self.max_depth): ###########
            self.best_feature = best_feature
            self.best_information_gain = best_information_gain
            if not terminal_node:
                # if leaf is on the right decision branch...
                if leaf_side:
                    # make the left decision branch a node
                    left_side_dataset = np.copy(self.xy_matrix[self.xy_matrix[:,best_feature] == False])
                    left = Node(
                        X=left_side_dataset[:,:-1],
                        Y=left_side_dataset[:,-1],
                        depth=self.depth+1,
                        max_depth=self.max_depth,
                        node_type='left_node'
                    )
                    right = Node(
                        node_type='leaf',
                        leaf_decision=leaf_decision,
                        leaf_side=leaf_side,
                    )
                    self.left = left
                    self.right = right
                    return self.left.grow_tree()
                    
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
                    left = Node(
                        node_type='leaf',
                        leaf_decision=leaf_decision,
                        leaf_side=leaf_side
                    )
                    self.left = left
                    self.right = right
                    return self.right.grow_tree()

            else:
                if leaf_side:
                    left = Node(
                        node_type='leaf',
                        leaf_decision=not leaf_decision,
                    )
                    right = Node(
                        node_type='leaf',
                        leaf_decision=leaf_decision,
                    )
                    self.left = left
                    self.right = right
                else:
                    left = Node(
                        node_type='leaf',
                        leaf_decision=leaf_decision,
                    )
                    right = Node(
                        node_type='leaf',
                        leaf_decision=not leaf_decision,
                    )
                    self.left = left
                    self.right = right
                return

    
    def print_format(self):
        
        if 'node' in self.node_type:
            print(self.node_type)
        else:
            leaf_side = 'right' if self.leaf_side else 'left'
            leaf_decision = 'yes' if self.leaf_decision else 'no'
            print(f'{leaf_side} leaf -> {leaf_decision}')


    def predict(self, xhat: np.array) -> bool:
        '''
        Prediction method that walks through
        decision tree recursively based on binary features 
        until it hits a leaf
        '''
        print(f'x hat value: {xhat[self.best_feature]}')
        if not xhat[self.best_feature]:
            # check if node
            if 'node' in self.left.node_type:
                return self.left.predict(xhat)
            else:
                if self.left.leaf_decision:
                    return True
                return False

            # check if node
        else:
            if 'node' in self.right.node_type:
                return self.right.predict(xhat)
            else:
                if self.right.leaf_decision:
                    return True
                return False

        #return True

def DT_train_binary(X: np.ndarray, Y: np.array, max_depth: int):
    _n = Node(X=X, Y=Y, max_depth=max_depth)
    _n.grow_tree()
    return _n

def DT_make_prediction(x: np.array, DT: Node) -> bool:
    return DT.predict(x)

def DT_test_binary(X: np.ndarray, Y: np.array, DT: Node):
    samples = X.shape[0]
    n_correct = 0
    for row in np.arange(samples):
        result = DT_make_prediction(X[row,:], DT)
        if result == Y[row]:
            n_correct += 1
    return (n_correct / samples)

if __name__ == "__main__":
    X, Y = read_features_labels('data_set_TV.txt')
    DT = DT_train_binary(X, Y, max_depth=-1)
    xhat = np.array([1,0,1,0])
    print(DT_test_binary(X,Y,DT))