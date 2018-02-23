from sklearn import datasets
digits = datasets.load_digits()


from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = svm.SVC(gamma=0.001, C=100.)
# clf = tree.DecisionTreeClassifier()
size = len(digits.data)
half = int(size/2)

features_train = digits.data[:half]
print("features_train")
print(features_train)
labels_train = digits.target[:half]
print("labels_train")
print(labels_train)
features_test = digits.data[half:]
print("features_test")
print(features_test)
labels_test = digits.target[half:]
print("labels_test")
print(labels_test)

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print('pred')
print(pred)
# pred = []
# for test in features_test:
#     pred.append(clf.predict(test))
print(accuracy_score(pred, labels_test))