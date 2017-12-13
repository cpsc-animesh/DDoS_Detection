'''
Created on 11-Oct-2017
@author: animesh
'''
from numpy import record
import numpy
import math
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import pandas
import csv,sys

filename = 'kddcup.data_10_percent'
filename_adj = 'my_file'

feature = ['duration', 'protocol_type', 'service', 'flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot' ,'num_failed_logins'
,'logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login'
,'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate'
,'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate'
,'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate', 'attack?']
total_features = (len(feature)-1)

#Open the adjusted KDD dataset set and store it in a 2D list and normalize the list
def normalize():
    print("Normalizing data..")
    #Read the KDD CSV file
    with open(filename_adj, 'r') as f:
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
        for j in range(0,41):
            traffic[i][j] = float(traffic[i][j])
            if(traffic[i][j]<=data_min):
                data_min = traffic[i][j]
            if(traffic[i][j]>data_max):
                data_max = traffic[i][j]

    for i in range(0, len(traffic)):
        for j in range(0,41):
            traffic[i][j] = float(traffic[i][j])
            traffic[i][j] = ((traffic[i][j])-data_min)/(data_max-data_min)
            traffic[i][j] = traffic[i][j] *1000
    print("Data Normalized")
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
        gain_list.append(gain(traffic, j, 41))
    gain_dict = dict(zip(feature, gain_list))
    print("The Information Gain values are:")
    print(gain_dict)
    print("The sorted Information Gain values are:")
    print(sorted(gain_dict, key = gain_dict.get))

def chi2_list():
    dataframe = pandas.read_csv(filename_adj, names=feature)
    #array = dataframe.values
    #X = dataframe.iloc[:,0:41]
    #Y = dataframe.iloc[:,41]
    array = dataframe.values
    X = array[:,0:41]
    Y = array[:,41]
    # feature extraction
    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(X,Y)
    chi2_list = fit.scores_
    chi2_dict = dict(zip(feature, chi2_list))
    print("The chi-squared values are: ")
    # summarize scores
    numpy.set_printoptions(precision=3)
    print(chi2_dict)
    print("The sorted chi-squared values are: ")
    print(sorted(chi2_dict, key = chi2_dict.get))

def main():
    print("Starting application..")
    traffic = normalize()
    print("1 - Display the entropy list")
    print("2 - Display the information gain list")
    print("3 - Display the chi-squared list")
    selection = input("Enter your selection: ")
    if(selection == 1):
        ent_list(traffic)
    elif(selection == 2):
        ig_list(traffic)
    elif(selection == 3):
        chi2_list()
    else:
        print("Invalid selection")
    
main()
    
    
    

