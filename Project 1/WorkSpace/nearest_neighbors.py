import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

def load_data(fname):
  f = open(fname, 'r')
  ctr = 0
  y_str = ''
  for line in f:
    line = line.strip().split(';')
    if ctr == 0:
      x_str = line
    else:
      y_str = line
    ctr+=1
  f.close()
  X = []
  Y = []
  for item in x_str:
    temp = [float(x) for x in item.split(',')]
    X.append(temp)
  if len(y_str)>0:
    for item in y_str:
      temp = int(item)
      Y.append(temp)
  X = np.array(X)
  Y = np.array(Y)
  return X, Y


class KNN:
    def __init__(
        self,
        X: np.ndarray,
        Y: np.array,
        K: int,
    ):
        self.Y = Y if Y is not None else np.zeros(shape=(2,4))# 1D array
        self.X = X if X is not None else np.zeros(shape=(2,4))# 2D array

        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]

        self.xy_matrix = np.column_stack((X,Y)) 
        self.K = K

        self.distances = None

    def distances_from_point(self, xhat: np.array) -> np.array:
        '''
        calculates euclidean distance between
        xhat and all points in training set
        '''
        samples = self.X.shape[0]
        distances = []
        for row in np.arange(samples):
            euclidian = spatial.distance.euclidean(xhat, self.X[row,:])
            distances.append((euclidian, self.Y[row]))
        self.distances = distances
        return distances

    def predict(self, xhat: np.ndarray) -> bool :
        '''
        predicts which label corresponds 
        with the data
        '''
        # sort distance measurements 
        # from desc to asc
        summation = 0
        distances = sorted(self.distances_from_point(xhat))
        #print(distances)
        for i in np.arange(self.K):
            summation += distances[i][1]
            print("Summation",summation)
            print(distances[i])
            print('\n\n')
        if summation > 0:
            return 1
        return -1

def KNN_test(X_train,Y_train,X_test,Y_test,K) -> float:
    knn = KNN(X_train, Y_train, K)
    knn_test = KNN(X_test, Y_test, K)
    samples = Y_test.shape[0]
    results = []
    for row in np.arange(samples):
        print('row ', row + 1)
        results.append(knn.predict(X_test[row,:]))
    compare = results == Y_test
    return (sum(compare) / len(compare))
    
