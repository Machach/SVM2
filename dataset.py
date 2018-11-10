import csv
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
import xlwt
df = pd.read_csv('E:\PycharmProjects\SVM2\datasets\Yeast\yeast-train.csv')


X_train = df.iloc[0:1000,:103]
Y_train = df.iloc[0:1000,103:117]
X_test = df.iloc[1000:,:103]
Y_test = df.iloc[1000:,103:117]
# print(Y_train)
X_traindata = pd.DataFrame(X_train)
Y_traindata = pd.DataFrame(Y_train)
X_testdata = pd.DataFrame(X_test)




#
# def iniClass():
#     for i in range(14):
#         for j in range(i,14):
#             class



def tt(alist,a,b):#创建二分类分类结果
    for i in range(1000):
        if Y_traindata.ix[i][a]==Y_traindata.ix[i][b]:
            alist.append('nan')
        elif Y_traindata.ix[i][a]>Y_traindata.ix[i][b]:
            alist.append(1)
        else:
            alist.append(0)

#训练
# list=[]
# tt(list,0,1)
# clf=LinearSVC()
# print(list)
# clf.fit(X=X_traindata,y=list)
# testy=clf.predict(X_testdata)
# print(testy)


file=xlwt.Workbook(encoding='utf-8')
table1=file.add_sheet('二分类器',cell_overwrite_ok=True)

m=0
for i in range(14):
    for j in range(i+1,14):
        list=[]
        list.append("class%d-class%d"%(i,j))
        tt(list,i,j)
        print(list)
        for k in range(1001):
            table1.write(k,m,list[k])
        m=m+1

file.save(r'E:\PycharmProjects\SVM2\22class.xls')












