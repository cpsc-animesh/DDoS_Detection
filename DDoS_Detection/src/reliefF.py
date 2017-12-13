from __future__ import print_function
import numpy as np
from sklearn.neighbors import KDTree
import csv, sys
import pandas

feature = ['duration', 'protocol_type', 'service', 'flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot' ,'num_failed_logins'
,'logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login'
,'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate'
,'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate'
,'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate', 'attack?']

class ReliefF(object):

    """Feature selection using data-mined expert knowledge.

    Based on the ReliefF algorithm as introduced in:

    Kononenko, Igor et al. Overcoming the myopia of inductive learning algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55

    """

    def __init__(self, n_neighbors=10, n_features_to_keep=10):
        """Sets up ReliefF to perform feature selection.

        Parameters
        ----------
        n_neighbors: int (default: 100)
            The number of neighbors to consider when assigning feature importance scores.
            More neighbors results in more accurate scores, but takes longer.

        Returns
        -------
        None

        """
        
        self.feature_scores = None
        self.top_features = None
        self.tree = None
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep

    
    def fit(self, X, y):
        """Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        None

        """
        self.feature_scores = np.zeros(36)
        self.tree = KDTree(X)
        #The shape attribute for numpy arrays returns the dimensions of the array. 
        #If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n
        for source_index in range(X.shape[0]):
            distances, indices = self.tree.query(X[source_index].reshape(1, -1), k=self.n_neighbors + 1)
            #Distances returns the distances of top 10 nearest neighbors from the row in question
            #Indices returns the location of those those 10 nearest neighbors
          
            # First match is self, so ignore it
            for neighbor_index in indices[0][1:]:
                similar_features = X[source_index] == X[neighbor_index]
                label_match = y[source_index] == y[neighbor_index]
                
                # If the labels match, then increment features that match and decrement features that do not match
                # Do the opposite if the labels do not match
                if label_match:
                    self.feature_scores[similar_features] += 1.
                    self.feature_scores[~similar_features] -= 1.
                else:
                    self.feature_scores[~similar_features] += 1.
                    self.feature_scores[similar_features] -= 1.
        self.top_features = np.argsort(self.feature_scores)[::-1]
        reliefF_list = self.feature_scores
        reliefF_dict = dict(zip(feature, reliefF_list))
        print("The reliefF values are:")
        print(reliefF_dict)
        print("The sorted reliefF values are:")
        print(sorted(reliefF_dict, key = reliefF_dict.get))
        #print(self.feature_scores)
        #print(self.top_features)
    
    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        return X[:, self.top_features[:self.n_features_to_keep]]

    def fit_transform(self, X, y):
        """Computes the feature importance scores from the training data, then reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        self.fit(X, y)
        return self.transform(X)
    
filename = 'kddcup.data_10_percent'
filename_adj = 'my_file'
dataframe = pandas.read_csv(filename_adj, names=feature)
array = dataframe.values
X = array[:,0:41]
Y = array[:,41]
obj = ReliefF()
obj.fit_transform(X, Y)
print("This is the last line!")






