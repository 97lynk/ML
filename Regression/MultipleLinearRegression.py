import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from Regression.Util import convertSize, removePlus, randomOutput
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# đọc file dataset_for_multi_linear_regression.csv gồm 3 feature - l label
dataset = pd.read_csv('dataset_for_multi_linear_regression.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values

# xử lý dữ liệu
converter_1 = np.vectorize(lambda str: convertSize(str))
converter_2 = np.vectorize(lambda str: removePlus(str))

X[:, 2] = converter_1(X[:, 2])  # column file size
y = converter_2(y)  # column labels

# chia dataset 20% cho testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('================================================')
########## Sử  dụng Linear Regression bình thường ##########
# sử dụng Linear Regression để training data
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# predict
y_pred = regr.predict(X_test)

# các hệ số và intercept
print('Linear Regression with three dependent variables')
print('Intercept: \n', regr.coef_)
print('Coefficients: \n', regr.coef_)
print('\ty = %+.2f  %+.2f * X_1  %+.2f * X_2  %+.2f * X_3 \n' % (
    regr.intercept_, regr.coef_[0], regr.coef_[1], regr.coef_[2]))
# calculate MAE, MSE, RMSE
print('Evaluate model:')
print('\tMAE = %.2f' % mean_absolute_error(y_test, y_pred))
print('\tMSE = %.2f' % mean_squared_error(y_test, y_pred))
print('\tRMSE = %.2f' % np.sqrt(mean_squared_error(y_test, y_pred)))
print('\tVariance score 1: %.2f' % r2_score(y_test, y_pred))
print('\tVariance score 2: %.2f' % regr.score(X_test, y_test))
print('================================================')

########## Sử dụng Linear Regression với random training data và LDA=1 ##########
lda = LDA(n_components=1)
X = lda.fit_transform(X, y)

# để biến phụ thuộc (output) thực tế hơn
randomOutput(y)

# chia dataset 20% cho testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# sử dụng Linear Regression để training data
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# predict
y_pred = regr.predict(X_test)

# các hệ số và intercept
print('Linear Regression with random training data and LDA')
print('Intercept: \n', regr.coef_)
print('Coefficients: \n', regr.coef_)
print('\ty = %+.2f  %+.2f * X\n' % (regr.intercept_, regr.coef_[0]))
# calculate MAE, MSE, RMSE
print('Evaluate model:')
print('\tMAE = %.2f' % mean_absolute_error(y_test, y_pred))
print('\tMSE = %.2f' % mean_squared_error(y_test, y_pred))
print('\tRMSE = %.2f' % np.sqrt(mean_squared_error(y_test, y_pred)))
print('\tVariance score 1: %.2f' % r2_score(y_test, y_pred))
print('\tVariance score 2: %.2f' % regr.score(X_test, y_test))
print('================================================')

# Vẽ
plt.figure(figsize=(6, 4))
plt.scatter(X_test, y_test, c='g', marker='+', label='Actual data')
plt.xlabel('LDA 1 (from Rating,Reviews,Size)')
plt.ylabel('Installs')
plt.plot(X_test, y_pred, c='r', label='Predicted data')
plt.legend(loc='best')
plt.title('Linear Regression with random training data and LDA')
plt.tight_layout()
plt.show()
