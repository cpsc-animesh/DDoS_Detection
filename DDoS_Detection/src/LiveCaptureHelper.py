'''
Created on Apr 3, 2018

@author: animesh
'''
from LiveCapture import *

sorted_IG_list = ['land', 'urgent', 'wrong_fragment', 'rerror_rate', 'srv_rerror_rate', 'count', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'duration', 'flag', 'serror_rate', 'srv_serror_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'diff_srv_rate', 'same_srv_rate', 'dst_host_same_srv_rate', 'srv_diff_host_rate', 'dst_host_diff_srv_rate', 'dst_host_srv_count', 'dst_host_srv_diff_host_rate', 'dst_host_count', 'protocol_type', 'srv_count', 'dst_host_same_src_port_rate', 'dst_bytes', 'service', 'src_bytes']
sorted_Chi2_list  = ['dst_host_same_srv_rate', 'count', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'wrong_fragment', 'dst_host_rerror_rate', 'diff_srv_rate', 'dst_host_srv_rerror_rate', 'dst_host_diff_srv_rate', 'serror_rate', 'srv_serror_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'flag', 'dst_host_same_src_port_rate', 'protocol_type', 'srv_diff_host_rate', 'dst_host_srv_diff_host_rate', 'dst_host_srv_count', 'service', 'land', 'urgent', 'dst_host_count', 'srv_count', 'duration', 'src_bytes', 'dst_bytes']
sorted_reliefF_list = ['srv_count', 'src_bytes', 'dst_bytes', 'dst_host_count', 'srv_diff_host_rate', 'dst_host_srv_diff_host_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_count', 'dst_host_diff_srv_rate', 'dst_host_same_srv_rate', 'same_srv_rate', 'diff_srv_rate', 'duration', 'dst_host_srv_rerror_rate', 'dst_host_rerror_rate', 'dst_host_srv_serror_rate', 'dst_host_serror_rate', 'service', 'serror_rate', 'srv_serror_rate', 'srv_rerror_rate', 'rerror_rate', 'flag', 'count', 'protocol_type', 'land', 'urgent', 'wrong_fragment']

def IG_NB(window_dataframe):
    header_list = window_dataframe.columns.values
    NumFeatures = 24
    selected_features = sorted_IG_list[0:NumFeatures]
    #Preparing the subset of the live window based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = window_dataframe[header_list[j]]
    window_array = df_clax.values
    # Loading the saved pickle
    model_pkl = open('IG_NB.pkl', 'rb')
    model = pickle.load(model_pkl)
    
    testSet_values = window_array.tolist()
    predictions = model.predict(testSet_values)
    print("The predicted values for IG_NB are: ")
    print(predictions)
    
def IG_SVM(window_dataframe):
    header_list = window_dataframe.columns.values
    NumFeatures = 23
    selected_features = sorted_IG_list[0:NumFeatures]
    #Preparing the subset of the live window based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = window_dataframe[header_list[j]]
    window_array = df_clax.values
    # Loading the saved pickle
    model_pkl = open('IG_SVM.pkl', 'rb')
    model = pickle.load(model_pkl)
    
    testSet_values = window_array.tolist()
    predictions = model.predict(testSet_values)
    print("The predicted values for IG_SVM are: ")
    print(predictions)
    
def IG_DT(window_dataframe):
    header_list = window_dataframe.columns.values
    NumFeatures = 23
    selected_features = sorted_IG_list[0:NumFeatures]
    #Preparing the subset of the live window based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = window_dataframe[header_list[j]]
    window_array = df_clax.values
    # Loading the saved pickle
    model_pkl = open('IG_DT.pkl', 'rb')
    model = pickle.load(model_pkl)
    
    testSet_values = window_array.tolist()
    predictions = model.predict(testSet_values)
    print("The predicted values for IG_DT are: ")
    print(predictions)
    
def IG_RF(window_dataframe):
    header_list = window_dataframe.columns.values
    NumFeatures = 23
    selected_features = sorted_IG_list[0:NumFeatures]
    #Preparing the subset of the live window based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = window_dataframe[header_list[j]]
    window_array = df_clax.values
    # Loading the saved pickle
    model_pkl = open('IG_RF.pkl', 'rb')
    model = pickle.load(model_pkl)
    
    testSet_values = window_array.tolist()
    predictions = model.predict(testSet_values)
    print("The predicted values for IG_RF are: ")
    print(predictions)



def Chi2_NB(window_dataframe):
    header_list = window_dataframe.columns.values
    NumFeatures = 25
    selected_features = sorted_Chi2_list[0:NumFeatures]
    #Preparing the subset of the live window based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = window_dataframe[header_list[j]]
    window_array = df_clax.values
    # Loading the saved pickle
    model_pkl = open('Chi2_NB.pkl', 'rb')
    model = pickle.load(model_pkl)
    
    testSet_values = window_array.tolist()
    predictions = model.predict(testSet_values)
    print("The predicted values for Chi2_NB are: ")
    print(predictions)

def Chi2_SVM(window_dataframe):
    header_list = window_dataframe.columns.values
    NumFeatures = 17
    selected_features = sorted_Chi2_list[0:NumFeatures]
    #Preparing the subset of the live window based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = window_dataframe[header_list[j]]
    window_array = df_clax.values
    # Loading the saved pickle
    model_pkl = open('Chi2_SVM.pkl', 'rb')
    model = pickle.load(model_pkl)
    
    testSet_values = window_array.tolist()
    predictions = model.predict(testSet_values)
    print("The predicted values for Chi2_SVM are: ")
    print(predictions)

def Chi2_DT(window_dataframe):
    header_list = window_dataframe.columns.values
    NumFeatures = 16
    selected_features = sorted_Chi2_list[0:NumFeatures]
    #Preparing the subset of the live window based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = window_dataframe[header_list[j]]
    window_array = df_clax.values
    # Loading the saved pickle
    model_pkl = open('Chi2_DT.pkl', 'rb')
    model = pickle.load(model_pkl)
    
    testSet_values = window_array.tolist()
    predictions = model.predict(testSet_values)
    print("The predicted values for Chi2_DT are: ")
    print(predictions)

def Chi2_RF(window_dataframe):
    header_list = window_dataframe.columns.values
    NumFeatures = 16
    selected_features = sorted_Chi2_list[0:NumFeatures]
    #Preparing the subset of the live window based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = window_dataframe[header_list[j]]
    window_array = df_clax.values
    # Loading the saved pickle
    model_pkl = open('Chi2_RF.pkl', 'rb')
    model = pickle.load(model_pkl)
    
    testSet_values = window_array.tolist()
    predictions = model.predict(testSet_values)
    print("The predicted values for Chi2_RF are: ")
    print(predictions)



def ReliefF_NB(window_dataframe):
    header_list = window_dataframe.columns.values
    NumFeatures = 18
    selected_features = sorted_reliefF_list[0:NumFeatures]
    #Preparing the subset of the live window based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = window_dataframe[header_list[j]]
    window_array = df_clax.values
    # Loading the saved pickle
    model_pkl = open('ReliefF_NB.pkl', 'rb')
    model = pickle.load(model_pkl)
    
    testSet_values = window_array.tolist()
    predictions = model.predict(testSet_values)
    print("The predicted values for ReliefF_NB are: ")
    print(predictions)
    
def ReliefF_SVM(window_dataframe):
    header_list = window_dataframe.columns.values
    NumFeatures = 1
    selected_features = sorted_reliefF_list[0:NumFeatures]
    #Preparing the subset of the live window based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = window_dataframe[header_list[j]]
    window_array = df_clax.values
    # Loading the saved pickle
    model_pkl = open('ReliefF_SVM.pkl', 'rb')
    model = pickle.load(model_pkl)
    
    testSet_values = window_array.tolist()
    predictions = model.predict(testSet_values)
    print("The predicted values for ReliefF_SVM are: ")
    print(predictions)

def ReliefF_DT(window_dataframe):
    header_list = window_dataframe.columns.values
    NumFeatures = 4
    selected_features = sorted_reliefF_list[0:NumFeatures]
    #Preparing the subset of the live window based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = window_dataframe[header_list[j]]
    window_array = df_clax.values
    # Loading the saved pickle
    model_pkl = open('ReliefF_DT.pkl', 'rb')
    model = pickle.load(model_pkl)
    
    testSet_values = window_array.tolist()
    predictions = model.predict(testSet_values)
    print("The predicted values for ReliefF_DT are: ")
    print(predictions)

def ReliefF_RF(window_dataframe):
    header_list = window_dataframe.columns.values
    NumFeatures = 18
    selected_features = sorted_reliefF_list[0:NumFeatures]
    #Preparing the subset of the live window based on the selected features
    df_clax = pandas.DataFrame(columns=selected_features)
    for i in range(len(selected_features)):
        for j in range(len(header_list)):
            if(header_list[j]==selected_features[i]):
                df_clax[selected_features[i]] = window_dataframe[header_list[j]]
    window_array = df_clax.values
    # Loading the saved pickle
    model_pkl = open('ReliefF_RF.pkl', 'rb')
    model = pickle.load(model_pkl)
    
    testSet_values = window_array.tolist()
    predictions = model.predict(testSet_values)
    print("The predicted values for ReliefF_RF are: ")
    print(predictions)

    
