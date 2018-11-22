import pandas as pd
from sklearn.linear_model import LinearRegression
import xlwt

df = pd.read_csv('E:\PycharmProjects\SVM2\datasets\Yeast\yeast-train.csv')

X_train = df.iloc[0:1000,:103]
Y_train = df.iloc[0:1000,103:117]
X_test = df.iloc[1000:,:103]
Y_test = df.iloc[1000:,103:117]


piaoshu = []

for i in range(1000):
    m = 0
    for j in range(14):
        m=m+Y_train.iat[i,j]
    piaoshu.append(m)

print(piaoshu)

linreg = LinearRegression()

linreg.fit(X_train,piaoshu)

print(linreg.intercept_)
print(linreg.coef_)

y_pred = linreg.predict(X_test)
# print('y_pred:',y_pred)
y_predint= [round(y_pred[i]) for i in range(500)]
print(y_predint)  ## 截取前n个标签，n取整
