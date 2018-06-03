import warnings
warnings.simplefilter("ignore")
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Open file and extract data
h5 = h5py.File('mias-mammography/all_mias_scans.h5', 'r')

data_bg_tissue = h5['BG']
data_class = h5['CLASS']

data_scan = h5['scan'][:][:, ::4, ::4]
data_scan_flat = data_scan.reshape((data_scan.shape[0], -1))

print(len(data_scan_flat[2]))


# Preprocess data for sklearn
class_le = LabelEncoder()
class_le.fit(data_class)
class_vec = class_le.transform(data_class)
bin_class_vec = []
for c in class_vec:
	if c == 5:
		bin_class_vec.append(1)
	else:
		bin_class_vec.append(0)


# Split dataset into testing and training
idx_vec = np.arange(data_scan_flat.shape[0])
x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
	data_scan_flat, class_vec, idx_vec, random_state = 2017, test_size = 0.3, stratify = class_vec)

bin_x_train, bin_x_test, bin_y_train, bin_y_test, bin_idx_train, bin_idx_test = train_test_split(
	data_scan_flat, bin_class_vec, idx_vec, random_state = 2017, test_size = 0.3, stratify = bin_class_vec)


# Machine learning
def train_and_test(clf):
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	score = 100*accuracy_score(y_test, y_pred)
	print('Accuracy %2.2f%%' % score)
	# creport = lambda gt_vec,pred_vec: classification_report(
	# 	gt_vec, pred_vec, target_names = [x.decode() for x in class_le.classes_])
	# print(creport(y_test, y_pred))
	return score

def bin_train_and_test(clf):
	clf.fit(bin_x_train, bin_y_train)
	bin_y_pred = clf.predict(bin_x_test)
	score = 100*accuracy_score(bin_y_test, bin_y_pred)
	# print('Binary Accuracy %2.2f%%' % score)
	return score


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(8, weights='distance'),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=0.0001, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()]

bin_classifiers = [
    KNeighborsClassifier(41, weights='distance'),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=0.0001, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()]

results = []

for i in range(len(classifiers)):
	print(names[i])
	results.append(train_and_test(classifiers[i]))
	bin_train_and_test(bin_classifiers[i])
	print()

for i in range(len(names)):
	print(names[i], results[i])

plt.bar(names, results)
plt.ylim((0, 100))
plt.xlabel("ML Algorithm")
plt.ylabel("Accuracy")
plt.title("ML Algorithm Accuracy")

for i,j in zip(np.arange(len(names)), results):
    plt.annotate("{:4.2f}%".format(j), xy=(i-0.3,j))

plt.show()
