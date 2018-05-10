print("Importing libraries...")
import warnings
warnings.simplefilter("ignore")
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


# Open file and extract data
print("Extracting data...")
h5 = h5py.File('mias-mammography/all_mias_scans.h5', 'r')

data_bg_tissue = h5['BG']
data_class = h5['CLASS']
data_scan = h5['scan'][:][:, ::16, ::16]
data_scan_flat = data_scan.reshape((data_scan.shape[0], -1))


# Preprocess data for sklearn
print("Preprocessing data...")
class_le = LabelEncoder()
class_le.fit(data_class)
class_vec = class_le.transform(data_class)


# Split dataset into testing and training
idx_vec = np.arange(data_scan_flat.shape[0])
x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
	data_scan_flat, class_vec, idx_vec, random_state = 2017, test_size = 0.5, stratify = class_vec)

# Machine learning
clf = MLPClassifier()
print("Training classifier...")
clf.fit(x_train, y_train)
print("Predicting...")
y_pred = clf.predict(x_test)
print('Accuracy %2.2f%%' % (100*accuracy_score(y_test, y_pred)))


# with h5py.File('mias-mammography/all_mias_scans.h5', 'r') as scan_h5:
#     bg_info = scan_h5['BG'][:]
#     class_info = scan_h5['CLASS'][:]
#     # low res scans
#     scan_lr = scan_h5['scan'][:][:, ::16, ::16]

# scan_lr_flat = scan_lr.reshape((scan_lr.shape[0], -1))


# from sklearn.preprocessing import LabelEncoder
# class_le = LabelEncoder()
# class_le.fit(class_info)
# class_vec = class_le.transform(class_info)
# class_le.classes_


# from sklearn.model_selection import train_test_split
# idx_vec = np.arange(scan_lr_flat.shape[0])
# x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(scan_lr_flat, 
#                                                     class_vec, 
#                                                     idx_vec,
#                                                     random_state = 2017,
#                                                    test_size = 0.5,
#                                                    stratify = class_vec)
# print('Training', x_train.shape)
# print('Testing', x_test.shape)


# # useful tools
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# creport = lambda gt_vec,pred_vec: classification_report(gt_vec, pred_vec, 
#                                                         target_names = [x.decode() for x in 
#                                                                         class_le.classes_])


# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(8)
# knn.fit(x_train, y_train)
# y_pred = knn.predict(x_test)
# print('Accuracy %2.2f%%' % (100*accuracy_score(y_test, y_pred)))
# print(creport(y_test, y_pred))