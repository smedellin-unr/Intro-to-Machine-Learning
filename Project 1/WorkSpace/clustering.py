import numpy as np
from scipy import spatial

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

class KMeansClustering:
    def __init__(
        self,
        X: np.ndarray,
        K: int,
        mu: np.ndarray,
        max_iter: int = 1000,
        ):
            self.X = X
            self.K = K

            self.n_samples = X.shape[0]
            self.n_features = X.shape[1]
            self.max_iter = max_iter

            self.mu_arr = mu if type(mu) is np.ndarray else self.random_mu_initialization()
            self.previous_mu_arr = None
    
    def random_mu_initialization(self) -> np.array:
        # Randomly chooses existing points
        # to be the means
        mus = np.empty((self.K, self.n_features))
        for i in np.arange(self.K):
            random_num = np.random.choice(range(self.n_samples), replace=False)
            mus[i] = self.X[random_num]
        return mus

    def create_clusters(self) -> list:
        # calculate distance between
        # point in question to the mus
        #
        # Place in appropriate cluster
        # intialize the correct number of clusters within a list
        clusters = [[] for c in range(self.K)]
        for row in np.arange(self.n_samples):
            distances = np.empty((0,self.n_samples),float)
            for k in np.arange(self.K):
                distances = np.append(distances,spatial.distance.euclidean(self.X[row], self.mu_arr[k]))
            min_val = np.amin(distances)
            min_index = np.where(distances == min_val)[0][0]
            clusters[min_index].append(self.X[row])
        return np.asarray(clusters)

    def calculate_new_mu(self):
        # generates new means based off of the 
        # clustering until the convergence or
        # until the max_iter limit is reached

        num_of_iter = 0
        while((not np.array_equal(self.mu_arr,self.previous_mu_arr)) and (num_of_iter <= self.max_iter)):
            self.previous_mu_arr = np.copy(self.mu_arr)
            clusters = self.create_clusters()
            for k in range(self.K):
                self.mu_arr[k] = np.sum(np.array(clusters[k]),axis=0) / 4
            num_of_iter+=1

def K_Means(X, K, mu):
    kmc = KMeansClustering(X, K, mu)
    return kmc.create_clusters()

