import pandas
import csv,sys
# from StdSuites.Table_Suite import row

filename = 'kddcup.data_10_percent'

feature = ['duration', 'protocol_type', 'service', 'flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot' ,'num_failed_logins'
,'logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login'
,'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate'
,'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate'
,'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate', 'attack?']

def find_distinct(attr):
    output = []
    for i in attr:
        if i not in output:
            output.append(i)
    return output

dataframe = pandas.read_csv(filename, names=feature)
array = dataframe.values
             
for i in range(1,4):
    Y = array[:,i]
    result = find_distinct(dataframe.iloc[:,i]) 
    print(result)
    num = 1
    for item in result:
        for k in range(len(Y)):
            if(Y[k]==item):
                Y[k] = num
        num = num+1
    print(Y)
    array[:,i] = Y
    print("##################")
print(array[116790])
dataframe2 = pandas.DataFrame(array)
dataframe2.to_csv('my_file')

