import pandas as pd
import matplotlib.pyplot as plt
import xlwt
import numpy as np
from sklearn import svm
from sklearn.preprocessing import Imputer

df1 = pd.read_csv('E:\PycharmProjects\SVM2\datasets\Yeast\yeast-train.csv')
X_train = df1.iloc[0:1000,:103]
X_test = df1._iloc[1001:,:103]
imp = Imputer(missing_values='NaN',strategy='mean',axis=1)


df2 = pd.read_excel('E:/PycharmProjects/SVM2/222class.xls')
Y_train = df2.iloc[0:1001,1:2]

result = pd.merge(X_train,Y_train,on=None,left_index=True,right_index=True)
result=result.dropna(axis=0)


X_train1=result.iloc[0:1000,:103]
Y_train1=result.iloc[0:1000,103:]
#
# print(X_train1)
# print(Y_train1)

# X_train=X_train.drop(axis=0,inplace=False)
# Y_train=Y_train.dropna(axis=0)
#

clf = svm.SVR()




clf.fit(X_train1,Y_train1.astype('int'))


Y_test = clf.predict(X_test)
print('result:',Y_test)


