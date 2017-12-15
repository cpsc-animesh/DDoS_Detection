from sklearn.naive_bayes import  GaussianNB
import pandas
import numpy as np

filename = 'kddcup.data_10_percent'

feature = ['duration', 'protocol_type', 'service', 'flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot' ,'num_failed_logins'
,'logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login'
,'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate'
,'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate'
,'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate', 'attack?']

dataframe = pandas.read_csv(filename, names=feature)
packets = dataframe.values
header_list = dataframe.columns.values
print(header_list)

selected_features = ['is_host_login','num_outbound_cmds','su_attempted','urgent','num_shells',
                      'land','root_shell','num_failed_logins','num_file_creations','num_access_files','num_root','attack?']

df_clax = pandas.DataFrame(columns=selected_features)
for i in range(len(selected_features)):
    for j in range(len(header_list)):
        if(header_list[j]==selected_features[i]):
            df_clax[selected_features[i]] = dataframe[header_list[j]]
array_clx = df_clax.values
X = array_clx[:,0:11]
print(X)

Y = array_clx[:,11]
print("#################")
print(Y)
model = GaussianNB()
model.fit(X,Y)

list1 = [['1','1','1','1','1','1','1','1','1','1','1'],
         ['0','0','0','0','0','0','0','0','0','0','0']]

for i in range(0, len(list1)):
    for j in range(0, len(list1[i])):
        list1[i][j] = float(list1[i][j])

print(list1)
predicted = model.predict(list1)
print(predicted)

df_clax.to_csv('file_clx', ',')
