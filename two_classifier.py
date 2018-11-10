
from sklearn import svm,datasets
#调用SVC()
clf = svm.SVC()
#载入鸢尾花数据集
iris = datasets.load_iris()
print(iris)