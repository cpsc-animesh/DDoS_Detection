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

filename_adj = 'chopped_my_file'
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
    #Read the live csv and pass it to the create_window function to create a window of 100 packets and classify the packets
    for i in range(5):
        with open(file_to_read, 'r') as f:
            read_file = csv.reader(f)
            window_dataframe = create_window(read_file, i)
            
            #Now get the selected FS and Classification algorithm from the GUI and classify this window
            sorted_reliefF_list = ['srv_count', 'src_bytes', 'dst_bytes', 'dst_host_count', 'srv_diff_host_rate', 'dst_host_srv_diff_host_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_count', 'dst_host_diff_srv_rate', 'dst_host_same_srv_rate', 'same_srv_rate', 'diff_srv_rate', 'duration', 'dst_host_srv_rerror_rate', 'dst_host_rerror_rate', 'dst_host_srv_serror_rate', 'dst_host_serror_rate', 'service', 'serror_rate', 'srv_serror_rate', 'srv_rerror_rate', 'rerror_rate', 'flag', 'count', 'protocol_type', 'land', 'urgent', 'wrong_fragment']
            selected_features = sorted_reliefF_list[0:16]
            selected_features.append('attack?')
            header_list = window_dataframe.columns.values
                
            #Preparing the subset of the live window based on the selected features
            df_clax = pandas.DataFrame(columns=selected_features)
            for i in range(len(selected_features)):
                for j in range(len(header_list)):
                    if(header_list[j]==selected_features[i]):
                        df_clax[selected_features[i]] = window_dataframe[header_list[j]]
            window_array = df_clax.values
            window_array = window_array[1:]
            window_array = window_array[:,:16]
            print("Window array first row: ")
            print(window_array[0])
            print(len(window_array[0]))
            print("*******************")
            
            #Reading the entire dataset
#             dataframe = pandas.read_csv(filename_adj, names=feature)
#             print("Not normalized dataset:")
#             print(dataframe.head(2))
#             print("***********************")
            traffic = normalize(filename_adj)
            dataframe = pandas.DataFrame(traffic)
            dataframe.columns = feature
            
            header_list = dataframe.columns.values
            #Preparing the subset of the dataset based on the selected features
            df_clax = pandas.DataFrame(columns=selected_features)
            for i in range(len(selected_features)):
                for j in range(len(header_list)):
                    if(header_list[j]==selected_features[i]):
                        df_clax[selected_features[i]] = dataframe[header_list[j]]
            dataframe_array = df_clax.values
            dataframe_array = dataframe_array[1:]
            print("Dataframe array first row: ")
            print(dataframe_array[0])
            print(len(dataframe_array[0]))
            
            trainSet, testSet = splitDataset(dataframe_array, 0.67)
            trainSet_part1 = []
            trainSet_part2 = []
            for packet in trainSet:
                trainSet_part1.append(packet[0:16])
                trainSet_part2.append(packet[16])
            
#             model = tree.DecisionTreeClassifier()
#             model = GaussianNB()
#             C_range = np.logspace(-1, 10, 1)
#             gamma_range = np.logspace(-1, 3, 1)
#             param_grid = dict(gamma=gamma_range, C=C_range)
#             model = GridSearchCV(SVC(), param_grid=param_grid)
            model = RandomForestClassifier(n_estimators=10)
            model.fit(trainSet_part1,trainSet_part2)
            testSet_values = window_array.tolist()
            predictions = model.predict(testSet_values)
            print("The predicted values are: ")
            print(predictions)
            print("##################################################################################")
            
            
if __name__ == "__main__":
    main()
