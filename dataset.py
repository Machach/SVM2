import csv
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier

df=pd.read_csv('E:\python test\SVM2\datasets\Yeast\yeast-train.csv')


X_train=df.iloc[:,:103]
Y_train=df.iloc[:,103:117]
print(Y_train)
X_traindata=pd.DataFrame(X_train)

Y_traindata=pd.DataFrame(Y_train)



def iniClass():
    for i in range(14):
        Class












Class12=[]
for i in range(1500):
    if Y_traindata.ix[i][0]==Y_traindata.ix[i][1]:
        Class12.append('')
    elif Y_traindata.ix[i][0]>Y_traindata.ix[i][1]:
        Class12.append(1)
    else:
        Class12.append(0)


print(Class12)



clf=LinearSVC()
clf.fit(X_traindata,Class12)





