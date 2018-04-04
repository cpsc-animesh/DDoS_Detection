'''
Created on Jan 23, 2018

@author: animesh
'''
import LiveCaptureHelper
from flask.globals import request

'''
This file continously reads from a rolling window of live network packets and filters out the features based on the selected
feature selection algorithm and applies the selected classification algorithm to return the prediction with an accuracy
percentage
'''

import csv
import pandas
from Training import *
import pickle
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

PATH = os.getcwd()
app = Flask(__name__)    
CORS(app)

feature = ['duration', 'protocol_type', 'service', 'flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate'
,'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate'
,'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']

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

def create_window(i):
    print("Creating window number:",i)
    fileContent=pandas.read_csv('/home/animesh/Documents/kdd99_feature_extractor-master/build/src/dump.csv', names = feature)
    start = (100*i) + 1
    end = (i+1) * 100
    hundred_lines = fileContent[start:end+1]
    hundred_lines_array = hundred_lines.values
#     print(len(hundred_lines_array[0]))
    cleaned_window = clean_window(hundred_lines_array)
#     print("The cleaned window is:")
#     print(cleaned_window[0])
    normalized_window = normalize_window(cleaned_window)
#     print("The normalized window is:")
#     print(len(normalized_window[0]))
    normalized_window = pandas.DataFrame(normalized_window)
    normalized_window.columns = feature
#     print("Normalized Window: ", normalized_window)
#     print("########################################################################################")
    return normalized_window


@app.route("/calculate", methods=["GET", "POST"])
def result():
    data = request.form.to_dict()
#     stopdata = int(request.form['input1'])
    featureAlgorithm = data['featureAlgorithm']
    classification = data['classification']
    main(featureAlgorithm, classification)
    print("This is after executing the main function")
    print(featureAlgorithm)
    print(classification)
    data = {'data': {'firstname': 'asdasd', 'lastname': 'utyuyt'}}
    return jsonify({'result': data})
   
   
#Dynamically read the newest 100 entries from the real_data CSV file, normalize it and store it in a data set
def main(featureAlgorithm, classification):
    print("Inside the main")
    featureAlgorithm = int(featureAlgorithm)
    classification = int(classification)

    #Read the live CSV and pass it to the create_window function to create a window of 100 packets and classify the packets
    i=0
    while 1:
        window_dataframe = create_window(i)    
      
        if(featureAlgorithm == 1 and classification == 1):
            LiveCaptureHelper.IG_NB(window_dataframe)
        elif(featureAlgorithm == 1 and classification == 2):
            LiveCaptureHelper.IG_SVM(window_dataframe)   
        elif(featureAlgorithm == 1 and classification == 3):
            LiveCaptureHelper.IG_DT(window_dataframe)            
        elif(featureAlgorithm == 1 and classification == 4):
            LiveCaptureHelper.IG_RF(window_dataframe)

        elif(featureAlgorithm == 2 and classification == 1):
            LiveCaptureHelper.Chi2_NB(window_dataframe)
        elif(featureAlgorithm == 2 and classification == 2):
            LiveCaptureHelper.Chi2_SVM(window_dataframe)
        elif(featureAlgorithm == 2 and classification == 3):
            LiveCaptureHelper.Chi2_DT(window_dataframe)
        elif(featureAlgorithm == 2 and classification == 4):
            LiveCaptureHelper.Chi2_RF(window_dataframe)
        
        elif(featureAlgorithm == 3 and classification == 1):
            LiveCaptureHelper.ReliefF_NB(window_dataframe)
        elif(featureAlgorithm == 3 and classification == 2):
            LiveCaptureHelper.ReliefF_SVM(window_dataframe)
        elif(featureAlgorithm == 3 and classification == 3):
            LiveCaptureHelper.ReliefF_DT(window_dataframe)
        elif(featureAlgorithm == 3 and classification == 4):
            LiveCaptureHelper.ReliefF_RF(window_dataframe)
            
        else:
            print("invalid Selection")
        i = i+1
if __name__ == "__main__":
    app.run()
