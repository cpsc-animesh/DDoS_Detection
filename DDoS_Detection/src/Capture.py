'''
Created on Jan 23, 2018

@author: animesh
'''

'''
This file continously reads from a rolling window of live network packets and filters out the features based on the selected
feature selection algorithm and applies the selected classification algorithm to return the prediction with an accuracy
percentage
'''

import csv,sys
import pandas

window_size = 100


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

def create_window(read_file, iter_no):
    window = []
    lines_to_skip = 100*iter_no
    read_file = read_file.readlines()[lines_to_skip:]
    for packet in read_file:
        if(len(window) == 100):
            break
        else:
            window.append(packet)
    print(window[0])
    cleaned_window = clean_window(window)
    normalized_window = normalize_window(cleaned_window)
    return normalized_window
            
#Dynamically read the newest 100 enteries from the real_data csv file, normalize it and store it in a dataset
def main():
    #Read the csv and pass it to the create_window function to create a window of 100 packets
    for i in range(1):
        with open(file_to_read, 'r') as f:
            read_file = csv.reader(f)
            created_window = create_window(read_file,0)
            #Now get the selected FS and Classification algorithm from the GUI and classify this window
            

if __name__ == "__main__":
    main()
