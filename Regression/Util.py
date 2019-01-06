from random import randrange

import numpy as np

# chuyển file size dạng string sang float với đơn vị kilobytes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


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

# random install
# ví dụ
# 1+ => trong khoảng 1 đến 4
# 5+ => trong khoảng 5 đến 9
# 10+ => trong khoảng 10 đến 49
# 50+ => trong khoảng 50 đến 99
# ...
def randomOutput(y):
    yy = np.unique(y)
    yy.sort()
    yy = yy[::-1]
    for j, v in enumerate(y):
        start = 1.0
        end = 1.0
        for index, el in enumerate(yy):
            if el <= v:
                if index != 0:
                    end = yy[index - 1]
                else:
                    end = 1.0
                start = el
                break

        if start == max(yy):
            end += start + randrange(0, 10000)
        if end == start:
            start = 1
            end = 5
        y[j] = randrange(start, end) * 1.0


def find(value, y):
    y = np.unique(y)
    y.sort()
    print('find %f' % value)
    for i in np.arange(1 ,len(y)):
        if y[i -1] <= value and value < y[i]:
            print(y[i -1])
            return y[i -1]

# tính array r^2, array degree cho từng cột data
# và tìm dataset (đã split) tốt nhất
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
