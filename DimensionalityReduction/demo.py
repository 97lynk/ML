import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('dataset_for_PCA_LDA.csv')
X = dataset.iloc[:, 0:39].values
y = dataset.iloc[:, 39].values

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

lb_enc = LabelEncoder()
y = lb_enc.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=0)

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1, store_covariance=True)
lda.fit_transform(X_std, y)
matrix_w = lda.scalings_

X_train, X_test =  X_train.dot(matrix_w), X_test.dot(matrix_w)

clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
y_predict = lb_enc.inverse_transform(y_predict)

import matplotlib.pyplot as plt

argsort_X_test = np.argsort(X_test[:, 0].ravel())
X_test_sorted = X_test[argsort_X_test, 0]
y_pred_sorted = y_predict[argsort_X_test]

def model(x):
    return 1 / (1 + np.exp(-x))


loss = model(X_test_sorted * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test_sorted, loss, c='r', label='Logistic model')
plt.scatter(X_test, lb_enc.inverse_transform(y_test), c='b', label='Actual datapoint')
plt.ylabel('Deffective')
plt.xlabel('LD1')
plt.title('Explained variance by different principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()