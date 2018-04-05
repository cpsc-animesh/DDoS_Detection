'''
Created on 11-Oct-2017
@author: animesh
There is always a trade off between bias and variance therefore find the sweet spot to increase the predictions
from the algorithms used
'''


from __future__ import print_function
from numpy import record
import numpy as np
import math
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KDTree
from sklearn.naive_bayes import  GaussianNB
from sklearn import svm
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import pandas
import csv,sys
import random
import xlwt
import pickle

filename = 'kddcup.data_10_percent'
# filename_adj = 'chopped_my_file'
filename_adj = 'my_file'

feature = ['duration', 'protocol_type', 'service', 'flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate'
,'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate'
,'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate', 'attack?']
total_features = (len(feature)-1)

#Open the adjusted KDD dataset set and store it in a 2D list and normalize the list
def normalize(file_to_normalize):
    print("Normalizing data..")
    #Read the KDD CSV file
    with open(file_to_normalize, 'r') as f:
        file = csv.reader(f)
        traffic = []
        try:
            for packet in file:
                packet = packet[1:]
                traffic.append(packet)
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (filename, file.line_num, e))
    f.close()
    #Normalizing the data
    data_min = 0
    data_max = 0
    for i in range(0, len(traffic)):
        for j in range(0,28):
            traffic[i][j] = float(traffic[i][j])
            if(traffic[i][j]<=data_min):
                data_min = traffic[i][j]
            if(traffic[i][j]>data_max):
                data_max = traffic[i][j]

    for i in range(0, len(traffic)):
        for j in range(0,28):
            traffic[i][j] = float(traffic[i][j])
            traffic[i][j] = ((traffic[i][j])-data_min)/(data_max-data_min)
            traffic[i][j] = traffic[i][j] *1000
    print("Data Normalized.\n")
    return traffic
                  
#Calculates the entropy of the given data set for the target attribute.
def entropy(data, target_attr):
    val_freq = {}
    data_entropy = 0.0
    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (val_freq.has_key(record[target_attr])):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]]  = 1.0
            
    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2)   
    return data_entropy

# Calculates the information gain (reduction in entropy) that would result by splitting the data on the chosen attribute (attr).
#target_attr = last attribute which tells if there is an attack or no attack
def gain(data, attr, target_attr):
    val_freq = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if (val_freq.has_key(record[attr])):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]]  = 1.0
          
    # Calculate the sum of the entropy for each subset of records weighted by their probability of occurence in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)
    # Subtract the entropy of the chosen attribute from the entropy of the whole data set with respect to the target attribute (and return it)
    return (entropy(data, target_attr) - subset_entropy)

def ent_list(traffic):
    ent_list=list()
    for i in range(total_features):
        ent_list.append(entropy(traffic,i))
    ent_dict = dict(zip(feature, ent_list))
    print("The entropy values are:")
    print(ent_dict)
    #Sort the entropy values
    print("The sorted entropy values are:")
    print(sorted(ent_dict, key = ent_dict.get))

def ig_list(traffic):
    gain_list=list()
    for j in range(total_features):
        gain_list.append(gain(traffic, j, 28))
    gain_dict = dict(zip(feature, gain_list))
    print("The Information Gain values are:")
    print(gain_dict)
    print("The sorted Information Gain values are:")
    sorted_gain_list = sorted(gain_dict, key = gain_dict.get)
    print(sorted_gain_list)
    return sorted_gain_list

def chi2_list():
    dataframe = pandas.read_csv(filename_adj, names=feature)
    #array = dataframe.values
    #X = dataframe.iloc[:,0:28]
    #Y = dataframe.iloc[:,28]
    array = dataframe.values
    X = array[:,0:28]
    Y = array[:,28]
    # feature extraction
    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(X,Y)
    chi2_list = fit.scores_
    chi2_dict = dict(zip(feature, chi2_list))
    print("The chi-squared values are: ")
    # summarize scores
    np.set_printoptions(precision=3)
    print(chi2_dict)
    print("The sorted chi-squared values are: ")
    sorted_chi2_list = sorted(chi2_dict, key = chi2_dict.get)
    print(sorted_chi2_list)
    return sorted_chi2_list

class reliefF(object):
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
        self.feature_scores = np.zeros(28)
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
        sorted_reliefF = sorted(reliefF_dict, key = reliefF_dict.get)
        print(sorted_reliefF)
        return sorted_reliefF
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
        sorted_reliefF_list = self.fit(X, y)
        return sorted_reliefF_list
        #return self.transform(X)

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = []
    copy = dataset.tolist()
    while(len(trainSet)<trainSize):
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def getAccuracy(testSet, predictions):
    correct = 0
    for packet in range(len(testSet)):
        if testSet[packet][-1] == predictions[packet]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
    
def naive_bayes(sorted_list, num_features):
    #Creating a list of the features based on the number of features to be selected
    selected_features = sorted_list[0:num_features]
    selected_features.append('attack?')
    print("Selected Features:")
    print(selected_features)
    
    #Reading the entire dataset
    dataframe = pandas.read_csv(filename_adj, names=feature)
    header_list = dataframe.columns.values
    
    #Preparing the subset of the dataset based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = dataframe[header_list[j]]
    array_clx = df_clax.values
    array_clx = array_clx[1:]

    #k-fold cross validation
    kf = KFold(n_splits=20) # Define the split - into 20 folds 
    sum = 0
    for train, test in kf.split(array_clx):
        train_data = np.array(array_clx)[train]
        test_data = np.array(array_clx)[test]
        train_data_part1 = []
        train_data_part2 = []
        for packet in train_data:
            train_data_part1.append(packet[0:num_features])
            train_data_part2.append(packet[num_features])
        model = GaussianNB()
        model.fit(train_data_part1,train_data_part2)
        testSet_values = []    
        for packet in test_data:
            testSet_values.append(packet[0:num_features])
        predictions = model.predict(testSet_values)
        accuracy = getAccuracy(test_data, predictions)
        sum += accuracy
    
    # Dump the trained decision tree classifier with Pickle
    NB_pkl_filename = 'ReliefF_NB.pkl'
    # Open the file to save as pkl file
    NB_model_pkl = open(NB_pkl_filename, 'w')
    pickle.dump(model, NB_model_pkl)
    # Close the pickle instances
    NB_model_pkl.close()

    #Find the average of the accuracy results
    average = sum/20
    print(average)
    return average

def svm(sorted_list, num_features):
    #Creating a list of the features based on the number of features to be selected
    selected_features = sorted_list[0:num_features]
    selected_features.append('attack?')
    print("Selected Features:")
    print(selected_features)
    
    #Reading the entire dataset
    dataframe = pandas.read_csv(filename_adj, names=feature)
    header_list = dataframe.columns.values
    
    #Preparing the subset of the dataset based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = dataframe[header_list[j]]
    array_clx = df_clax.values
    array_clx = array_clx[1:]
     
    #k-fold cross validation
    kf = KFold(n_splits=5) # Define the split - into 20 folds 
    sum = 0
    for train, test in kf.split(array_clx):
        train_data = np.array(array_clx)[train]
        test_data = np.array(array_clx)[test]
        train_data_part1 = []
        train_data_part2 = []
        for packet in train_data:
            train_data_part1.append(packet[0:num_features])
            train_data_part2.append(packet[num_features])
        #Optimizing the parameters(C, gamma) of SVM and using those parameter values to fit the model
        C_range = np.logspace(-1, 10, 1)
        gamma_range = np.logspace(-1, 3, 1)
        param_grid = dict(gamma=gamma_range, C=C_range)
        model = GridSearchCV(SVC(), param_grid=param_grid)
        model.fit(train_data_part1,train_data_part2)
        testSet_values = []    
        for packet in test_data:
            testSet_values.append(packet[0:num_features])
        predictions = model.predict(testSet_values)
        accuracy = getAccuracy(test_data, predictions)
        sum += accuracy
    
    # Dump the trained decision tree classifier with Pickle
    SVM_pkl_filename = 'ReliefF_SVM.pkl'
    # Open the file to save as pkl file
    SVM_model_pkl = open(SVM_pkl_filename, 'w')
    pickle.dump(model, SVM_model_pkl)
    # Close the pickle instances
    SVM_model_pkl.close()    
    
    #Find the average of the accuracy results
    average = sum/5
    print(average)
    return average

def decision_tree(sorted_list, num_features):
    #Creating a list of the features based on the number of features to be selected
    selected_features = sorted_list[0:num_features]
    selected_features.append('attack?')
    print("Selected Features:")
    print(selected_features)
    
    #Reading the entire dataset
    dataframe = pandas.read_csv(filename_adj, names=feature)
    header_list = dataframe.columns.values
    
    #Preparing the subset of the dataset based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = dataframe[header_list[j]]
    array_clx = df_clax.values
    array_clx = array_clx[1:]
    
    #k-fold cross validation
    kf = KFold(n_splits=20) # Define the split - into 20 folds 
    sum = 0
    for train, test in kf.split(array_clx):
        train_data = np.array(array_clx)[train]
        test_data = np.array(array_clx)[test]
        train_data_part1 = []
        train_data_part2 = []
        for packet in train_data:
            train_data_part1.append(packet[0:num_features])
            train_data_part2.append(packet[num_features])
        model = tree.DecisionTreeClassifier()
        model.fit(train_data_part1,train_data_part2)
        testSet_values = []    
        for packet in test_data:
            testSet_values.append(packet[0:num_features])
        predictions = model.predict(testSet_values)
        accuracy = getAccuracy(test_data, predictions)
        sum += accuracy
    
    DT_pkl_filename = 'ReliefF_DT.pkl'
    # Open the file to save as pkl file
    DT_model_pkl = open(DT_pkl_filename, 'w')
    pickle.dump(model, DT_model_pkl)
    # Close the pickle instances
    DT_model_pkl.close()    

    
    #Find the average of the accuracy results
    average = sum/20
    print(average)
    return average

def randomForest(sorted_list, num_features):
    #Creating a list of the features based on the number of features to be selected
    selected_features = sorted_list[0:num_features]
    selected_features.append('attack?')
    print("Selected Features:")
    print(selected_features)
    
    #Reading the entire dataset
    dataframe = pandas.read_csv(filename_adj, names=feature)
    header_list = dataframe.columns.values
    
    #Preparing the subset of the dataset based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = dataframe[header_list[j]]
    array_clx = df_clax.values
    array_clx = array_clx[1:]

    #k-fold cross validation
    kf = KFold(n_splits=20) # Define the split - into 20 folds 
    sum = 0
    for train, test in kf.split(array_clx):
        train_data = np.array(array_clx)[train]
        test_data = np.array(array_clx)[test]
        train_data_part1 = []
        train_data_part2 = []
        for packet in train_data:
            train_data_part1.append(packet[0:num_features])
            train_data_part2.append(packet[num_features])
        model = RandomForestClassifier(n_estimators=10)
        model.fit(train_data_part1,train_data_part2)
        testSet_values = []    
        for packet in test_data:
            testSet_values.append(packet[0:num_features])
        predictions = model.predict(testSet_values)
        accuracy = getAccuracy(test_data, predictions)
        sum += accuracy
    
    # Dump the trained decision tree classifier with Pickle
    RF_pkl_filename = 'ReliefF_RF.pkl'
    # Open the file to save as pkl file
    RF_model_pkl = open(RF_pkl_filename, 'w')
    pickle.dump(model, RF_model_pkl)
    # Close the pickle instances
    RF_model_pkl.close()
    #Find the average of the accuracy results
    
    average = sum/20
    print(average)
    return average

def create_dataframe():
    dataframe = pandas.read_csv(filename_adj, names=feature)
    array = dataframe.values
    X = array[:,0:28]
    Y = array[:,28]
    return X,Y

def main():
    print("Starting application..\n")
    traffic = normalize(filename_adj)
    print("1 - Display the entropy list")
    print("2 - Display the information gain list")
    print("3 - Display the chi-squared list")
    print("4 - Display the ReliefF list")
    selection = input("Enter your selection: ")
    #num_features = input("Enter the number of features to select: ")
    if(selection == 1):
        ent_list(traffic)
    elif(selection == 2):
        #sorted_gain_list = ig_list(traffic)
        sorted_gain_list = ['land', 'urgent', 'wrong_fragment', 'rerror_rate', 'srv_rerror_rate', 'count', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'duration', 'flag', 'serror_rate', 'srv_serror_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'diff_srv_rate', 'same_srv_rate', 'dst_host_same_srv_rate', 'srv_diff_host_rate', 'dst_host_diff_srv_rate', 'dst_host_srv_count', 'dst_host_srv_diff_host_rate', 'dst_host_count', 'protocol_type', 'srv_count', 'dst_host_same_src_port_rate', 'dst_bytes', 'service', 'src_bytes']
        print("Features selected using Information Gain.")
    elif(selection == 3):
        #sorted_chi2_list = chi2_list()
        print("Features selected using Chi-squared.")
        sorted_chi2_list  = ['dst_host_same_srv_rate', 'count', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'wrong_fragment', 'dst_host_rerror_rate', 'diff_srv_rate', 'dst_host_srv_rerror_rate', 'dst_host_diff_srv_rate', 'serror_rate', 'srv_serror_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'flag', 'dst_host_same_src_port_rate', 'protocol_type', 'srv_diff_host_rate', 'dst_host_srv_diff_host_rate', 'dst_host_srv_count', 'service', 'land', 'urgent', 'dst_host_count', 'srv_count', 'duration', 'src_bytes', 'dst_bytes']
    elif(selection == 4):
        #X,Y = create_dataframe()
        #obj = reliefF()
        #sorted_reliefF_list = obj.fit_transform(X, Y)
        sorted_reliefF_list = ['srv_count', 'src_bytes', 'dst_bytes', 'dst_host_count', 'srv_diff_host_rate', 'dst_host_srv_diff_host_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_count', 'dst_host_diff_srv_rate', 'dst_host_same_srv_rate', 'same_srv_rate', 'diff_srv_rate', 'duration', 'dst_host_srv_rerror_rate', 'dst_host_rerror_rate', 'dst_host_srv_serror_rate', 'dst_host_serror_rate', 'service', 'serror_rate', 'srv_serror_rate', 'srv_rerror_rate', 'rerror_rate', 'flag', 'count', 'protocol_type', 'land', 'urgent', 'wrong_fragment']
        print("Features selected using reliefF.")
    else:
        print("Invalid selection")
    print("\n")
    print("*****************")
    print("Classification")
    print("*****************")
    print("1 - Naive Bayes ")
    print("2 - SVM")
    print("3 - Decision Tree")
    print("4 - Random Forest")
    clx_selection = input("Enter your selection: ")
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1")
    sheet1.write(0, 2, "Accuracy Values")
    for i in range(1,2):
        num_features = i
        print("Number of features selected - ", i)
        if(selection == 2 and clx_selection == 1):
            print("Classifying using Naive Bayes...")
            accuracy = naive_bayes(sorted_gain_list, num_features)
            sheet1.write(i, 2, accuracy)
        elif(selection == 3 and clx_selection == 1):
            print("Classifying using Naive Bayes...")
            accuracy = naive_bayes(sorted_chi2_list, num_features)
            sheet1.write(i, 5, accuracy)
        elif(selection == 4 and clx_selection == 1):
            print("Classifying using Naive Bayes...")
            accuracy = naive_bayes(sorted_reliefF_list, num_features)
            sheet1.write(i, 2, accuracy)
            
        elif(selection == 2 and clx_selection == 2):
            print("Classifying using SVM...")
            accuracy = svm(sorted_gain_list, num_features)
            sheet1.write(i, 3, accuracy)
        elif(selection == 3 and clx_selection == 2):
            print("Classifying using SVM...")
            accuracy = svm(sorted_chi2_list, num_features)
            sheet1.write(i, 3, accuracy)
        elif(selection == 4 and clx_selection == 2):
            print("Classifying using SVM...")
            accuracy = svm(sorted_reliefF_list, num_features)
            sheet1.write(i, 3, accuracy)
        
        elif(selection == 2 and clx_selection == 3):
            print("Classifying using Decision Trees...")
            accuracy = decision_tree(sorted_gain_list, num_features)
            sheet1.write(i, 3, accuracy)
        elif(selection == 3 and clx_selection == 3):
            print("Classifying using Decision Trees...")
            accuracy = decision_tree(sorted_chi2_list, num_features)
            sheet1.write(i, 3, accuracy)
        elif(selection == 4 and clx_selection == 3):
            print("Classifying using Decision Trees...")
            accuracy = decision_tree(sorted_reliefF_list, num_features)
            sheet1.write(i, 3, accuracy)
        
        elif(selection == 2 and clx_selection == 4):
            print("Classifying using Random Forest...")
            accuracy = randomForest(sorted_gain_list, num_features)
            sheet1.write(i, 6, accuracy)
        elif(selection == 3 and clx_selection == 4):
            print("Classifying using Random Forest...")
            accuracy = randomForest(sorted_chi2_list, num_features)
            sheet1.write(i, 6, accuracy)
        elif(selection == 4 and clx_selection == 4):
            print("Classifying using Random Forest...")
            accuracy = randomForest(sorted_reliefF_list, num_features)
            sheet1.write(i, 6, accuracy)    
        
        else:
            print("Invalid Selection")
        print("End")
    book.save("results.xls")

if __name__ == "__main__":
    main()
    
    
    

