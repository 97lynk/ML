# đọc file dataset_for_poly_regression.csv gồm 14 feature
import pandas as pd
dataset = pd.read_csv('dataset_for_poly_regression.csv')
X = dataset.iloc[:, 0:14].values
y = dataset.iloc[:, 14].values

# tính array r^2, array degree cho từng cột data
# và tìm dataset (đã split) tốt nhất
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

def findBestPolyLinearRegression(dataset, degrees):
    y = dataset.iloc[:, 14].values

    X_train, X_test, Y_train, Y_test = train_test_split(dataset.iloc[:, 0:1].values, y, test_size=0.2)
    best_feature_index, best_degree, best_r2 = 0, 1, 0
    # Mảng lưu giá trị các R^2
    r2_array = []
    degree_array = []

    for i in np.arange(0, 14):
        X = dataset.iloc[:, i:i + 1].values
        # Tạo data huấn luyện và test
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # R^2 lớn nhất
        # degree nhỏ nhất
        max_r2, min_deg = 0.0, 0
        for deg in degrees:
            # Tạo 1 feature matrix mới với degree = deg
            poly_model = Pipeline([('poly', PolynomialFeatures(degree=deg)),
                                   ('linear', LinearRegression(fit_intercept=False))])
            poly_model.fit(x_train, y_train)
            y_predict = poly_model.predict(x_test)

            # tính R^2
            r2 = r2_score(y_test, y_predict)

            # tìm min degree và max R^2
            if max_r2 <= r2:
                max_r2 = r2
                min_deg = deg

        # Lưu vào mảng
        r2_array.append(max_r2)
        degree_array.append(min_deg)

        if best_r2 < max_r2:
            best_feature_index, best_degree, best_r2 = i, min_deg, max_r2
            X_train, X_test, Y_train, Y_test = x_train, x_test, y_train, y_test

    return X_train, X_test, Y_train, Y_test, best_feature_index, degree_array, r2_array

# tìm từ bậc 2 đến 10
import numpy as np
degrees_range = np.arange(2, 11)

X_train, X_test, y_train, y_test, best_feature_index, degree_array, r2_array = \
    findBestPolyLinearRegression(dataset, degrees_range)

best_r2 = r2_array[best_feature_index]
best_degree = degree_array[best_feature_index]
best_column_name = dataset.columns[best_feature_index]

print('Best found:')
print('\tR^2 = %.2f' % best_r2)
print('\tdegree r^2 = %d' % best_degree)
print('\tcolumn = %s' % best_column_name)
print('=================')
for i in np.arange(0, 14):
    print('Column %s [R^2 = %.4f, Degree = %d]' % (dataset.columns[i], r2_array[i], degree_array[i]))

#vẽ số liệu về r2 và degree với từng cột
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
for i in np.arange(0, 14):
    plt.scatter(degree_array[i], r2_array[i], label=dataset.columns[i])
    plt.text(degree_array[i] + 0.1, round(r2_array[i], 2) - 0.01, '%.3f' % r2_array[i])
plt.xlabel('Degree')
plt.ylabel('R^2')
plt.xticks(degrees_range)
plt.yticks(degrees_range / 10.0)
plt.tight_layout()
plt.legend()
plt.show()

# tạo lại model với số cột phù hợp, degree tốt nhất và data tốt
model = Pipeline([('poly', PolynomialFeatures(degree=best_degree)),
                  ('linear', LinearRegression(fit_intercept=False))])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# in phương trình
eq = '\ty = '
for i, c in enumerate(model.get_params().get('linear').coef_):
    if i == 0:
        eq += ' %+.2f ' % c
    else:
        eq += ' %+.2fX^%d' % (c, i)
print(eq)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print('Evaluate model:')
print('\tMAE = %.2f' % mean_absolute_error(y_test, y_pred))
print('\tMSE = %.2f' % mean_squared_error(y_test, y_pred))
print('\tRMSE = %.2f' % np.sqrt(mean_squared_error(y_test, y_pred)))
print('\tVariance score: %f' % r2_score(y_test, y_pred))
print('================================================')

# vẽ data
plt.figure(2)
# sort từng cặp (X_test, y_predict) theo giá trị X_test
# để vẽ dường Poly Linear Regression đẹp hơn
argsort_X_test = np.argsort(X_test[:, 0].ravel())
X_test_sorted = X_test[argsort_X_test, 0]
y_pred_sorted = y_pred[argsort_X_test]

plt.scatter(X_test, y_test, label='Actual data')
plt.plot(X_test_sorted, y_pred_sorted, color='green', label='$' + eq + '$')
plt.title('Poly Linear Regression with %s' % best_column_name)
plt.xlabel(best_column_name)
plt.ylabel('Bodyfat')
plt.legend()
plt.show()