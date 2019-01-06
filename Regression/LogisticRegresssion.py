# đọc data từ file
import pandas as pd
dataset = pd.read_csv('dataset_for_logistic_regression.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score

# split data 20% cho testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train_best, X_test_best = X_train, X_test
max_accuracy, best_index = .0, 0
for i in range(0, 3):
    X_train_, X_test_ = X_train[:, i:i + 1], X_test[:, i:i + 1]
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X_train_, y_train)
    y_pred = clf.predict(X_test_)
    acc = accuracy_score(y_test, y_pred)
    print('%s - %f' % (dataset.columns[i], acc))
    if max_accuracy <= acc:
        max_accuracy = acc
        best_index = i
        X_train_best, X_test_best = X_train_, X_test_

print('Best found:')
print('\tColumns %s with accuracy = %f' % (dataset.columns[best_index], max_accuracy))

# sử dụng data đã split
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train_best, y_train)
y_pred = clf.predict(X_test_best)
print('Intercept: \n', clf.intercept_)
print('Coefficients: \n', clf.coef_)

# đánh giá model
import numpy as np
target_names = np.unique(y)
actual_names = ['actual %s' % c for c in target_names]
predict_names = ['predicted %s' % c for c in target_names]
conf_1 = confusion_matrix(y_test, y_pred, target_names)
table_conf_1 = pd.DataFrame(conf_1, actual_names, predict_names)
report_1 = classification_report(y_test, y_pred, target_names)
print('Using Logistic Regression')
print('===================Confusion Matrix 1==================\n%s' % table_conf_1)
print('===================Report 1============================\n%s' % report_1)

import numpy as np
import matplotlib.pyplot as plt

# hàm signmol
def model(x):
    return 1 / (1 + np.exp(-x))

# xếp xếp X_test và y_pred để vẽ không lộn xộn
argsort_X_test = np.argsort(X_test_best[:, 0].ravel())
X_test_sorted = X_test_best[argsort_X_test, 0]
y_pred_sorted = y_pred[argsort_X_test]

loss = model(X_test_sorted * clf.coef_ + clf.intercept_).ravel()
plt.figure(1)
plt.plot(X_test_sorted, loss, c='r', label='Logistic model')
plt.scatter(X_test_best, y_test, c='b', label='Actual datapoint')
plt.ylabel('Target')
plt.xlabel(dataset.columns[best_index])
plt.title('Logistic Regression with binary output')
plt.legend(loc='best')
plt.tight_layout()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# sử dụng LDA
lda = LDA(n_components=1)
X_new = lda.fit_transform(X, y)
matrix_w = lda.scalings_

# chiếu data cũ
X_train, X_test = X_train.dot(matrix_w), X_test.dot(matrix_w)

# sử dụng lại Logistic Regression
clf2 = LogisticRegression(solver='lbfgs')
clf2.fit(X_train, y_train)
y_predict = clf2.predict(X_test)

# in đánh giá
conf_2 = confusion_matrix(y_test, y_pred, target_names)
table_conf_2 = pd.DataFrame(conf_2, actual_names, predict_names)
report_2 = classification_report(y_test, y_pred, target_names)
print('Using Logistic Regression after apply LDA with n = 1')
print('===================Confusion Matrix 2==================\n%s' % table_conf_2)
print('===================Report 2============================\n%s' % report_2)

# xếp xếp X_test và y_pred để vẽ không lộn xộn
argsort_X_test = np.argsort(X_test[:, 0].ravel())
X_test_sorted = X_test[argsort_X_test, 0]
y_pred_sorted = y_pred[argsort_X_test]

loss = model(X_test_sorted * clf2.coef_ + clf2.intercept_).ravel()
plt.figure(2)
plt.plot(X_test_sorted, loss, c='r', label='Logistic model')
plt.scatter(X_test, y_test, c='b', label='Actual datapoint')
plt.ylabel('Target')
plt.xlabel('LD1')
plt.title('Logistic Regression with binary output 2')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

