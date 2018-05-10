from sklearn import datasets
iris = datasets.load_iris()


from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score

def preprocess():
    ftrain = []
    ltrain = []
    ftest = []
    ltest = []
    for i in range(len(iris.data)):
        if i % 2 == 0:
            ftrain.append(iris.data[i])
            ltrain.append(iris.target[i])
        else:
            ftest.append(iris.data[i])
            ltest.append(iris.target[i])
    return ftrain, ltrain, ftest, ltest

# clf = svm.SVC(gamma=0.001, C=100.)
clf = tree.DecisionTreeClassifier()

features_train, labels_train, features_test, labels_test = preprocess()

print("features_train")
print(features_train)
print("labels_train")
print(labels_train)
print("features_test")
print(features_test)
print("labels_test")
print(labels_test)

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print('pred')
print(pred)

print(accuracy_score(pred, labels_test))