import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss

def one_vs_rest(X,Y,x,y):
    model = OneVsRestClassifier(svm.SVC(kernel='linear'))
    clf = model.fit(X, Y)

    y_predit = clf.predict(x)

    print('一对多评价系数：', classification_report(y, y_predit))
    print('汉明损失：', hamming_loss(y, y_predit))


def one_vs_one(X,Y,x,y):
    model = OneVsOneClassifier(svm.SVC(kernel='linear'))
    clf = model.fit(X, Y)

    y_predit = clf.predict(x)

    print('一对一评价系数：', classification_report(y, y_predit))
    print('汉明损失：', hamming_loss(y, y_predit))

df = pd.read_csv('E:\PycharmProjects\SVM2\datasets\Yeast\yeast-train.csv')


X_train = df.iloc[0:1000,:103]
Y_train = df.iloc[0:1000,103:117]
X_test = df.iloc[1000:,:103]
Y_test = df.iloc[1000:,103:117]

# one_vs_one(X_train,Y_train,X_test,Y_test)
one_vs_rest(X_train,Y_train,X_test,Y_test)

