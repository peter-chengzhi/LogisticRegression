"""
This program performs two different logistic regression implementations on two
different datasets of the format [float,float,boolean], one
implementation is in this file and one from the sklearn library. The program
then compares the two implementations for how well the can predict the given outcome
for each input tuple in the datasets.

@author Per Harald Borgen
"""

import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

# scale larger positive and values to between -1,1 depending on the largest
# value in the data
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
df = pd.read_csv("formated.csv", header=0)

#特征
feature_array = []
wd_df = pd.read_csv("wd.csv", header=None, sep=',')
for i in range (wd_df.index.stop):
    if wd_df.loc[i][1] >= 250:
        feature_array.append(wd_df.loc[i][0])

columns = feature_array.copy()
columns.append('label')
df.columns = columns

x = df["label"].map(lambda x: float(x))
print(x)
# formats the input data into two arrays, one of independant variables
# and one of the dependant variable
X = df[feature_array]
X = np.array(X)
X = min_max_scaler.fit_transform(X)
Y = df["label"].map(lambda x: float(x))
Y = np.array(Y)

# if want to create a new clean dataset 
##X = pd.DataFrame.from_records(X,columns=['grade1','grade2'])
##X.insert(2,'label',Y)
##X.to_csv('data2.csv')

# creating testing and training set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33)

# train scikit learn model 
clf = LogisticRegression()
clf.fit(X_train, Y_train)
print ('score Scikit learn: ', clf.score(X_test, Y_test))

# visualize data, uncomment "show()" to run it
#pos = where(Y == 1)
#neg = where(Y == 0)
#scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
#scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
#xlabel('Exam 1 score')
#ylabel('Exam 2 score')
#legend(['Not Admitted', 'Admitted'])
#show()

def Test_LR():
    score = 0
    winner = ""
    #first scikit LR is tested for each independent var in the dataset and its prediction is compared against the dependent var
    #if the prediction is the same as the dataset measured value it counts as a point for thie scikit version of LR
    scikit_score = clf.score(X_test,Y_test)
    print("begin print weight")
    print(clf.coef_)
    print("scikit_score:", scikit_score)

Test_LR()
