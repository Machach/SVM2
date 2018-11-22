import pandas as pd
import matplotlib.pyplot as plt
import xlwt
import numpy as np
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss


def final_test(vote_dataframe,vote_sum,row,col,):  # 最终产生结果的方法，vote_dataframe为排好序的测试集，vote_sum为投票阈值的列表，row/col为测试集标签的规模
    for i in range(row):
        # votex = vote_sum[i]
        votex = 4
        for j in range(col):
            if vote_dataframe.iat[i,j] <= votex:
                vote_dataframe.iat[i,j] = 1
            else:
                vote_dataframe.iat[i,j] = 0


def train_sub_clf(X,y,X_test,df3,i):   # 训练子SVM分类器并预测
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(X, y)  # 训练完毕
    Y_test = clf.predict(X_test)
    df3[i]=Y_test


def vote(k,DataFrame):  # 投票环节，k为标签数目，DataFrame取值测试集的单独一行
    m=0
    vote_list=[0 for i in range(k)]
    for i in range(k):
        for j in range(i+1,k):
            if DataFrame.iat[0,m]==1:
                vote_list[i] = vote_list[i]+1
            else:
                vote_list[j] = vote_list[j]+1
            m = m+1
    return vote_list


def vote_regre(X_train,Y_train,X_test):  # 对测试集的票数前n位进行线性回归
    piaoshu = []
    for i in range(1000):
        m = 0
        for j in range(14):
            m = m + Y_train.iat[i, j]
        piaoshu.append(m)
    linreg = LinearRegression()
    linreg.fit(X_train, piaoshu)
    y_pred = linreg.predict(X_test)
    # print('y_pred:',y_pred)
    y_predint = [round(y_pred[i]) for i in range(500)]
    print('y_predint:',y_predint)
    return y_predint  # 截取前n个标签，n取整



# 准备训练样本
df1 = pd.read_csv('E:\PycharmProjects\SVM2\datasets\Yeast\yeast-train.csv')
X_train = df1.iloc[0:1000,:103]
Y_train = df1.iloc[0:1000:,103:]
# 准备测试样本的特征
X_test = df1._iloc[1000:,:103]
Y_test = df1.iloc[1000:,103:117]

y_predin = vote_regre(X_train,Y_train,X_test)

# 准备训练标签
df2 = pd.read_excel('E:/PycharmProjects/SVM2/12class.xls')
df3 = pd.DataFrame()
df4 = pd.DataFrame()

# 创建工作表存放二分类结果
file = xlwt.Workbook(encoding='utf-8')
table2 = file.add_sheet('二分类结果',cell_overwrite_ok=True)

for i in range(0,91):
    Y_train1 = df2.iloc[0:1001,i:i+1]
    result = pd.merge(X_train,Y_train1,on=None,left_index=True,right_index=True)  # 合并训练样本的特征向量与每一个子分类器的结果
    result = result.dropna(axis=0)  # 去除训练标签为空的样本
    X_train1 = result.iloc[0:1000,:103].values  # 分离特征向量与标签
    Y_train2 = result.iloc[0:1000,103:].values.ravel().astype('int')
    train_sub_clf(X_train1,Y_train2,X_test,df3,i)
print("分类完毕，准备投票")
# print('df3', df3)
# df3.to_excel(excel_writer='E:\\PycharmProjects\\SVM2\\result1.xls',sheet_name='haha')

for i in range(500):
    df4[i]=vote(14,df3.ix[i:i+1])

df4=df4.T.rank(axis=1,ascending=False)  # 根据票数降序排列，1代表票数最多。


final_test(df4,y_predin,500,14)


print('df4:',df4)


print(accuracy_score(y_true=Y_test,y_pred=df4))
print(classification_report(y_true=Y_test,y_pred=df4))
# print(confusion_matrix(y_true=Y_test,y_pred=df4))
print('汉明损失：',hamming_loss(y_true=Y_test,y_pred=df4))