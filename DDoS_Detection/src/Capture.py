'''
Created on Jan 23, 2018

@author: animesh
'''

'''
This file continously reads from a rolling window of live network packets and filters out the features based on the selected
feature selection algorithm and applies the selected classification algorithm to return the prediction with an accuracy
percentage
'''

import csv
import pandas
from main import *

feature = ['duration', 'protocol_type', 'service', 'flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate'
,'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate'
,'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate', 'attack?']

file_to_read = '/home/animesh/Documents/kdd99_feature_extractor-master/build/src/dump.csv'

def normalize_window(cleaned_window):
    traffic = cleaned_window
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
    return traffic
    
def find_distinct(attr):
    output = []
    for i in attr:
        if i not in output:
            output.append(i)
    return output

def clean_window(window):
    dataframe = pandas.DataFrame(window)
    array = dataframe.values
#     print(len(array[0]))
#     print(array[0])
    for i in range(1,4):
        Y = array[:,i]
        result = find_distinct(dataframe.iloc[:,i]) 
#         print("Distinct values in column",i,"are:")
#         print(result)
        num = 1
        for item in result:
            for k in range(len(Y)):
                if(Y[k]==item):
                    Y[k] = num
            num = num+1
#         print("Y:")
#         print(Y)
        array[:,i] = Y
    return array

def create_window(read_file, i):
    fileContent=pandas.read_csv('/home/animesh/Documents/kdd99_feature_extractor-master/build/src/dump.csv', names = feature)

    start = (100*i) + 1
    end = (i+1) * 100
#     print("Start at: ", start)
#     print("End at: ", end)
    hundred_lines = fileContent[start:end]
    hundred_lines_array = hundred_lines.values
#     print(hundred_lines_array)
#     print(len(hundred_lines))
    cleaned_window = clean_window(hundred_lines_array)
    normalized_window = normalize_window(cleaned_window)
    normalized_window = pandas.DataFrame(normalized_window)
    normalized_window.columns = feature
#     print("Normalized Window: ", normalized_window)
#     print("########################################################################################")
    return normalized_window
            
#Dynamically read the newest 100 enteries from the real_data csv file, normalize it and store it in a dataset
def main():
    #Read the csv and pass it to the create_window function to create a window of 100 packets
    for i in range(1):
        with open(file_to_read, 'r') as f:
            read_file = csv.reader(f)
            dataframe = create_window(read_file, i)
            #Now get the selected FS and Classification algorithm from the GUI and classify this window
            sorted_reliefF_list = ['srv_count', 'src_bytes', 'dst_bytes', 'dst_host_count', 'srv_diff_host_rate', 'dst_host_srv_diff_host_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_count', 'dst_host_diff_srv_rate', 'dst_host_same_srv_rate', 'same_srv_rate', 'diff_srv_rate', 'duration', 'dst_host_srv_rerror_rate', 'dst_host_rerror_rate', 'dst_host_srv_serror_rate', 'dst_host_serror_rate', 'service', 'serror_rate', 'srv_serror_rate', 'srv_rerror_rate', 'rerror_rate', 'flag', 'count', 'protocol_type', 'land', 'urgent', 'wrong_fragment', 'attack?']
            selected_features = sorted_reliefF_list[0:16]
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
                    train_data_part1.append(packet[0:15])
                    train_data_part2.append(packet[15])
                model = tree.DecisionTreeClassifier()
                model.fit(train_data_part1,train_data_part2)
                testSet_values = []    
                for packet in test_data:
                    testSet_values.append(packet[0:15])
                predictions = model.predict(testSet_values)
                accuracy = getAccuracy(test_data, predictions)
                sum += accuracy
            
            #Find the average of the accuracy results
            average = sum/20
            print(average)
            return average

            
            
            
            
            
            
if __name__ == "__main__":
    main()
