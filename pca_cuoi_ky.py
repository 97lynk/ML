import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# đọc file Wine.csv gồm 12 feature - l label
dataset = pd.read_csv('dataset_for_PCA_LDA.csv')
X = dataset.iloc[:, 0:39].values
y = dataset.iloc[:, 39].values

# Code phần 1: dùng bất kì giải thuật classification nào mà sinh viên đã học để phân loại (ví dụ giải thuật A)
# Giải thuật Navie Bayes
# gnb = GaussianNB()
# gnb = svm.SVC(kernel='linear')
gnb = KNeighborsClassifier(n_neighbors=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

gnb.fit(X_train, y_train)
y_predict = gnb.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

target_names = ['YES', 'NO']
print(classification_report(y_test, y_predict, target_names=target_names))
conf = confusion_matrix(y_test, y_predict, target_names);
actual_names = ['actual YES', 'actual NO']
predict_names = ['predicted YES', 'predicted NO']
print(pd.DataFrame(conf, actual_names, predict_names))

# chuẩn hóa data: trừ toàn bộ data cho vector mean
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA as sklearnPCA
pca = sklearnPCA(n_components = 2)
Y = pca.fit_transform(X_std)


# dùng lại giải thuật navie bayes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Y, y, test_size = 0.2, random_state = 0)

# gnb2 = GaussianNB()
# gnb2 = svm.SVC(kernel='linear')
gnb2 = KNeighborsClassifier(n_neighbors=1)

gnb2.fit(X_train, y_train)
y_predict = gnb2.predict(X_test);
target_names = ['YES', 'NO']
print(classification_report(y_test, y_predict, target_names=target_names))
# y_pred = [0, 0, 2, 2, 0, 2]

conf = confusion_matrix(y_test, y_predict, target_names);
actual_names = ['actual YES', 'actual NO']
predict_names = ['predicted YES', 'predicted NO']
print(pd.DataFrame(conf, actual_names, predict_names))

print(pca.get_covariance())