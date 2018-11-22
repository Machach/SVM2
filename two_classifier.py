
from sklearn import svm,datasets
from sklearn.multiclass import OneVsOneClassifier
#调用SVC()
clf = svm.SVC()
#载入鸢尾花数据集
iris = datasets.load_iris()
print(iris)

X, y = iris.data, iris.target
clf = svm.LinearSVC(random_state=0)

clf = OneVsOneClassifier(clf)  # 根据二分类器构建多分类器
clf.fit(X, y)  # 训练模型
y_pred = clf.predict(X) # 预测样本
print('预测正确的个数：%d,预测错误的个数：%d' %((y==y_pred).sum(),(y!=y_pred).sum()))