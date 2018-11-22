import csv
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
import xlwt
df = pd.read_csv('E:\PycharmProjects\SVM2\datasets\Yeast\yeast-train.csv')
from sklearn.preprocessing import MultiLabelBinarizer

X_train = df.iloc[0:1000,:103].values

Y_train = df.iloc[0:1000,103:117].values
X_test = df.iloc[1000:,:103].values
Y_test = df.iloc[1000:,103:117].values

X_traindata = pd.DataFrame(X_train)
Y_traindata = pd.DataFrame(Y_train)

X_testdata = pd.DataFrame(X_test)

clf=OneVsOneClassifier(estimator=SVC(random_state=0))
clf.fit(X_traindata,Y_traindata)