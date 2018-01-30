'''
Created on Jan 23, 2018

@author: animesh
'''

#from main.py import *
from DataCleaning import clean_data
from main import randomForest, normalize
import pandas
import random
from sklearn.ensemble import RandomForestClassifier
import csv

'''
This file continously reads from a rolling window of live network packets and filters out the features based on ReliefF
and applies decision tree classification to return the status of every packet\
'''
filename_adj = 'chopped_my_file'
filename = 'realdata.csv'
feature = ['duration', 'protocol_type', 'service', 'flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate'
           ,'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate'
           ,'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate', 'attack?']

sorted_chi2_list  = ['dst_host_same_srv_rate', 'count', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'wrong_fragment', 'dst_host_rerror_rate', 'diff_srv_rate', 'dst_host_srv_rerror_rate', 'dst_host_diff_srv_rate', 'serror_rate', 'srv_serror_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'flag', 'dst_host_same_src_port_rate', 'protocol_type', 'srv_diff_host_rate', 'dst_host_srv_diff_host_rate', 'dst_host_srv_count', 'service', 'land', 'urgent', 'dst_host_count', 'srv_count', 'duration', 'src_bytes', 'dst_bytes']
  
clean_data(filename, feature)
normalize('real_data_cleaned')

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = []
    copy = dataset.tolist()
    while(len(trainSet)<trainSize):
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def randomForest(sorted_list, num_features):
    selected_features = sorted_list[0:num_features]
    selected_features.append('attack?')
    print("Selected Features:")
    print(selected_features)
    
    dataframe = pandas.read_csv(filename_adj, names=feature)
    #packets = dataframe.values
    header_list = dataframe.columns.values
    
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = dataframe[header_list[j]]
    array_clx = df_clax.values
    array_clx = array_clx[1:]
    trainSet, testSet = splitDataset(array_clx, 0.67)
    trainSet_part1 = []
    trainSet_part2 = []
    for packet in trainSet:
        trainSet_part1.append(packet[0:num_features])
        trainSet_part2.append(packet[num_features])

    model = RandomForestClassifier(n_estimators=10)
    model.fit(trainSet_part1,trainSet_part2)
    with open('real_data_cleaned', 'r') as f:
        file = csv.reader(f)
        testSet_values = []
        for packet in file:
            testSet_values.append(packet)
   
    predictions = model.predict(testSet_values)
    print("The predictions values are:")
    print(predictions)
    accuracy = getAccuracy(testSet, predictions)
    print(accuracy)
    return accuracy

randomForest(sorted_chi2_list, 16)

