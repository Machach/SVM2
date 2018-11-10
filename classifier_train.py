import pandas as pd
import xlwt
import numpy as np
from sklearn import svm
from sklearn.preprocessing import Imputer

df1 = pd.read_csv('E:\PycharmProjects\SVM2\datasets\Yeast\yeast-train.csv')
X_train = df1.iloc[0:1000,:103]
X_test = df1._iloc[1001:,:103]
imp = Imputer(missing_values='NaN',strategy='mean',axis=1)


df2 = pd.read_excel('E:/PycharmProjects/SVM2/2class.xls')
Y_train = df2.iloc[1:1001,0:1].fillna(value=np.nan)
#
# print(Y_train)
Y_train1 = imp.transform(Y_train)
clf = svm.SVC()
print(imp.transform(Y_train))

clf.fit(X_train,imp.transform(Y_train).values.ravel())
Y_test = clf.predict(X_test)
print(Y_test)


