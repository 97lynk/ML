# đọc data từ file
import pandas as pd
dataset = pd.read_csv('dataset_for_PCA_LDA.csv')
X = dataset.iloc[:, 0:39].values
y = dataset.iloc[:, 39].values

# chuẩn hóa data về 0 -> 1
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

# encode label NO/YES -> 0/1
from sklearn.preprocessing import LabelEncoder
lb_enc = LabelEncoder()
y = lb_enc.fit_transform(y)

# split data 20% cho testing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=0)
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

# in report
target_names = ['YES', 'NO']
actual_names = ['actual YES', 'actual NO']
predict_names = ['predicted YES', 'predicted NO']
y_test_report, y_pred_report = lb_enc.inverse_transform(y_test), lb_enc.inverse_transform(y_predict)

from sklearn.metrics import confusion_matrix, classification_report
conf_1 = confusion_matrix(y_test_report, y_pred_report, target_names)
table_conf_1 = pd.DataFrame(conf_1, actual_names, predict_names)
report_1 = classification_report(y_test_report, y_pred_report, target_names)
print('Using Logistic Regression after apply LDA')
print('===================Confusion Matrix 1==================\n%s' % table_conf_1)
print('===================Report 1============================\n%s' % report_1)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1, store_covariance=True)
X_new = lda.fit_transform(X_std, y)
matrix_w = lda.scalings_
print('Eigenvector = LDA 1:\n%s' % matrix_w)

X_train, X_test = X_train.dot(matrix_w), X_test.dot(matrix_w)

clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
y_pred_report = lb_enc.inverse_transform(y_predict)

conf_2 = confusion_matrix(y_test_report, y_pred_report, target_names)
table_conf_2 = pd.DataFrame(conf_2, actual_names, predict_names)
report_2 = classification_report(y_test_report, y_pred_report, target_names)
print('Using Logistic Regression after apply LDA with n = 1')
print('===================Confusion Matrix 2==================\n%s' % table_conf_2)
print('===================Report 2============================\n%s' % report_2)

import matplotlib.pyplot as plt
import numpy as np

# hàm signmol
def model(x):
    return 1 / (1 + np.exp(-x))

# xếp xếp X_test và y_pred để vẽ không lộn xộn
argsort_X_test = np.argsort(X_test[:, 0].ravel())
X_test_sorted = X_test[argsort_X_test, 0]
y_pred_sorted = y_predict[argsort_X_test]

loss = model(X_test_sorted * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test_sorted, loss, c='r', label='Logistic model')
plt.scatter(X_test, lb_enc.inverse_transform(y_test), c='b', label='Actual datapoint')
plt.ylabel('Deffective')
plt.xlabel('LD1')
plt.title('Logistic Regression with binary output')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
