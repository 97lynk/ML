from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# đọc file dataset_for_poly_regression.csv gồm 14 feature - l label
from Regression.Util import findBestPolyLinearRegression

dataset = pd.read_csv('dataset_for_poly_regression.csv')
X = dataset.iloc[:, 0:14].values
y = dataset.iloc[:, 14].values

degrees_range = np.arange(2, 11)

### Tìm cột có R^2 cao nhất và degree luôn
# vì mỗi lần split data ngẫu nhiên (shuffle) ảnh hưởng số degree tốt nhất
# nên cầu lưu lại data đã split đó
X_train, X_test, y_train, y_test, best_feature_index, degree_array, r2_array = findBestPolyLinearRegression(dataset,
                                                                                                            degrees_range)
# Abdomen là index 6 với R^2 lớn thứ 2 thử vẽ nó xem
# best_feature_index = 6
# X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, best_feature_index:best_feature_index+1].values, y, test_size=0.2)

best_r2 = r2_array[best_feature_index]
best_degree = degree_array[best_feature_index]
best_column_name = dataset.columns[best_feature_index]

## tạo lại model với số cột phù hợp, degree tốt nhất và data tốt
model = Pipeline([('poly', PolynomialFeatures(degree=best_degree)),
                   ('linear', LinearRegression(fit_intercept=False))])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


## report và vẽ số liệu về r2 và degree cùng cột
plt.figure(figsize=(6, 4))
for i in np.arange(0, 14):
    print('\tCột %s [R^2 = %.4f, Degree = %d]' % (dataset.columns[i], r2_array[i], degree_array[i]))
    plt.scatter(degree_array[i], r2_array[i], label=dataset.columns[i])
    plt.text(degree_array[i] + 0.1, round(r2_array[i], 2) - 0.01, '%.3f' % r2_array[i])
print('\t==> Tốt nhất là cột %s [R^2 = %.4f và Degree = %d]'
      % (best_column_name, best_r2, best_degree))

plt.xlabel('Degree')
plt.ylabel('R^2')
plt.xticks(degrees_range)
plt.yticks(degrees_range / 10.0)
plt.tight_layout()
plt.legend()

eq = '\ty = '
for i, c in enumerate(model.get_params().get('linear').coef_):
    if i == 0:
        eq += ' %+.2f ' % c
    else:
        eq += ' %+.2fX^%d' % (c, i)
print(eq)

print('Evaluate model:')
print('\tMAE = %.2f' % mean_absolute_error(y_test, y_pred))
print('\tMSE = %.2f' % mean_squared_error(y_test, y_pred))
print('\tRMSE = %.2f' % np.sqrt(mean_squared_error(y_test, y_pred)))
print('\tVariance score: %f' % r2_score(y_test, y_pred))
print('================================================')

# vẽ data
plt.figure()

# sort từng cặp (X_test, y_predict) theo giá trị X_test
# để vẽ dường Poly Linear Regression đạp hơn
argsort_X_test = np.argsort(X_test[:,0].ravel())
X_test_sorted = X_test[argsort_X_test, 0]
y_pred_sorted = y_pred[argsort_X_test]

plt.scatter(X_test, y_test, label='Actual data')
plt.plot(X_test_sorted, y_pred_sorted, color='green',label='$' + eq + '$')
plt.title('Poly Linear Regression with %s' % best_column_name)
plt.xlabel(best_column_name)
plt.ylabel('Bodyfat')
plt.legend()
plt.tight_layout()
plt.show()

