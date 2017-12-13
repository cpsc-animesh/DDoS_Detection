'''
Created on 17-Oct-2017

@author: animesh
'''
'''
# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
print(X)
Y = array[:,8]
print(Y)
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])
'''


# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

filename = 'kddcup.data_10_percent'

feature = ['duration', 'protocol_type', 'service', 'flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot' ,'num_failed_logins'
,'logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login'
,'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate'
,'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate'
,'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate', 'attack?']

dataframe = pandas.read_csv(filename, names=feature)
#array = dataframe.values
X = dataframe.iloc[:,4:40]
Y = dataframe.iloc[:,41]

# feature extraction
test = SelectKBest(score_func=chi2, k=10)
fit = test.fit(X, Y)
chi2_list = fit.scores_
chi2_dict = dict(zip(feature, chi2_list))
print("The chi-squared values are: ")
# summarize scores
numpy.set_printoptions(precision=3)
print(chi2_dict)
print("The sorted chi-squared values are: ")
print(sorted(chi2_dict, key = chi2_dict.get))