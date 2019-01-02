from cProfile import label

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# đọc file dataset_for_poly_regression.csv gồm 14 feature - l label
dataset = pd.read_csv('dataset_for_poly_regression.csv')
X_ = dataset.iloc[:, 0:14].values
y = dataset.iloc[:, 14].values

# pca = PCA(n_components=1)
# X = pca.fit_transform(X)
# print(pca.explained_variance_ratio_)

# Khai báo các giá trị của degree là từ 1 đến 10
degrees = np.arange(0, 11)
plt.figure(figsize=(6, 6))
for i in np.arange(0, 14):
    X = dataset.iloc[:,i:i + 1].values
    # Tạo data huấn luyện và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Mảng lưu giá trị các RMES
    # RMSE là căn bậc hai của phương sai, và được gọi là độ lệch chuẩn,
    # tức là sự khác biệt giữa các ước lượng và những gì được đánh giá
    r2_array = []

    # Khai báo biến lưu giá trị RMSE nhỏ nhất, biến lưu giá trị degree nhỏ nhất
    max_r2, min_deg = 0.0, 0

    for deg in degrees:

        # Tạo 1 feature matrix mới với degree = deg
        poly_features = PolynomialFeatures(degree=deg + 1, include_bias=False)
        x_poly_train = poly_features.fit_transform(X_train)

        # Khai báo Linear regression
        poly_reg = LinearRegression()
        poly_reg.fit(x_poly_train, y_train)

        # So sánh với dữ kiệu test
        x_poly_test = poly_features.fit_transform(X_test)
        poly_predict = poly_reg.predict(x_poly_test)

        # Sai số toàn phương trung bình (Đối với một ước lượng không có thiên vị, MSE là phương sai của ước lượng)
        # poly_mse = mean_squared_error(y_test, poly_predict)
        # RMSE bằng căn bậc 2 của MSE
        # poly_rmse = np.sqrt(poly_mse)
        poly_r2 = r2_score(y_test, poly_predict)
        # print(poly_r2)

        # Lưu vào mảng
        r2_array.append(poly_r2)

        # Tìm dregree và RMSE
        if max_r2 < poly_r2:
            max_r2 = poly_r2
            min_deg = deg

    # In ra thông tin
    print('Degree tốt nhất là {} với R^2 = {} ở cột {}'.format(min_deg, max_r2, dataset.columns[i]))

    # Vẽ hình
    # ax = fig.add_subplot(111)
    plt.scatter(min_deg, max_r2, label=dataset.columns[i])
    plt.text(min_deg + 0.1, round(max_r2, 2) - 0.01, '%.3f'%max_r2)

plt.xlabel('Degree')
plt.ylabel('R^2')

plt.xticks(degrees)
plt.yticks(degrees/10.0)
plt.tight_layout()
plt.legend()
plt.show()
