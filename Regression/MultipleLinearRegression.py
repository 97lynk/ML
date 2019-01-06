import pandas as pd
dataset = pd.read_csv('dataset_for_multi_linear_regression.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values

def convertSize(str):
    if 'M' in str:
        return float(str.replace('M', '')) * 1024.0  # bỏ M và nhân với 1024.0
    elif 'k' in str:
        # print(str, float(str.replace('k', '')) * 1.0)
        return float(str.replace('k', '')) * 1.0  # chỉ bỏ k

# xóa đấu + và chuyển sang float
def removePlus(str):
    str = str.replace(',', '')
    return float(str.replace('+', ''))

# xử lý dữ liệu
import numpy as np
converter_1 = np.vectorize(lambda str: convertSize(str))
converter_2 = np.vectorize(lambda str: removePlus(str))

X[:, 2] = converter_1(X[:, 2])  # column file size
y = converter_2(y)  # column install

# chia dataset 20% cho testing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
X_new = pca.fit_transform(X)

matrix_w = pca.components_.T
X_train, X_test = X_train.dot(matrix_w), X_test.dot(matrix_w)

# sử dụng Linear Regression để training data
regr2 = LinearRegression()
regr2.fit(X_train, y_train)
y_pred = regr2.predict(X_test)


# các hệ số và intercept
print('================================================')
print('Linear Regression with random training data and PCA')
print('PCA explained variance ratio: %s' % pca.explained_variance_ratio_)
print('Intercept: \n', regr2.coef_)
print('Coefficients: \n', regr2.coef_)
print('\ty = %+.2f  %+.2f * X\n' % (regr2.intercept_, regr2.coef_[0]))
# calculate MAE, MSE, RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('Evaluate model:')
print('\tMAE = %.2f' % mean_absolute_error(y_test, y_pred))
print('\tMSE = %.2f' % mean_squared_error(y_test, y_pred))
print(f'\tRMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')
print('\tVariance score: %.2f' % r2_score(y_test, y_pred))
print('================================================')

# Vẽ
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.scatter(X_test, y_test, c='g', marker='+', label='Actual data')
plt.xlabel('PCA 1 (decomposition from Rating,Reviews,Size)')
plt.ylabel('Installs')
plt.plot(X_test, y_pred, c='r', label='Predicted data')
plt.legend(loc='best')
plt.title('Linear Regression with random training data and PCA')
plt.tight_layout()
plt.show()
