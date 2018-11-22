import pandas as pd
import xlwt

df = pd.read_csv('E:\PycharmProjects\SVM2\datasets\Yeast\yeast-train.csv')

X_train = df.iloc[0:1000,:103]
Y_train = df.iloc[0:1000,103:117]
X_test = df.iloc[1000:,:103]
Y_test = df.iloc[1000:,103:117]



def tt(alist,a,b):  # 创建二分类分类结果
    for i in range(1000):
        if Y_train.ix[i][a] == Y_train.ix[i][b]:
            alist.append('nan')
        elif Y_train.ix[i][a] > Y_train.ix[i][b]:
            alist.append(1)
        else:
            alist.append(-1)

file = xlwt.Workbook(encoding='utf-8')
table1 = file.add_sheet('二分类器',cell_overwrite_ok=True)

m = 0
for i in range(14):
    for j in range(i+1,14):
        temp = []
        temp.append("class%d-class%d"%(i+1,j+1))
        tt(temp,i,j)
        print(temp)
        for k in range(1001):
            table1.write(k,m,temp[k])
        m = m+1

file.save(r'E:\PycharmProjects\SVM2\12class.xls')












