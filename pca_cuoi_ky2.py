import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# đọc file Wine.csv gồm 12 feature - l label
dataset = pd.read_csv('dataset_for_PCA_LDA.csv')
X = dataset.iloc[:, 0:39].values
y = dataset.iloc[:, 39].values

# Code phần 1: dùng bất kì giải thuật classification nào mà sinh viên đã học để phân loại (ví dụ giải thuật A)
# Giải thuật Navie Bayes
gnb = GaussianNB()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

gnb.fit(X_train, y_train)

y_predict = gnb.predict(X_test);


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

target_names = ['YES', 'NO']
print(classification_report(y_test, y_predict, target_names=target_names))
# y_pred = [0, 0, 2, 2, 0, 2]

conf = confusion_matrix(y_test, y_predict, target_names);
actual_names = ['actual YES', 'actual NO']
predict_names = ['predicted YES', 'predicted NO']
print(pd.DataFrame(conf, actual_names, predict_names))

####################

# chuẩn hóa data: trừ toàn bộ data cho vector mean
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

cov_mat = np.cov(X_std.T)
print('Ma trận hiệp phương sai: \n%s' % cov_mat)

mean_vec = np.mean(X_std, axis=0)
cov_mat2 = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Ma trận hiệp phương sai 2: \n%s' % cov_mat2)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

cor_mat1 = np.corrcoef(X_std.T)
print('Ma trận hệ số tương quan: \n%s' % cor_mat1)
#
# eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
# # print('Eigenvectors \n%s' %eig_vecs)
# # print('\nEigenvalues \n%s' %eig_vals)
#

cor_mat2 = np.corrcoef(X.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
u, s, v = np.linalg.svd(X_std.T)

print(eig_vecs)
print(eig_vecs)
# for ev in eig_vecs:
#     np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=10)
# print('Everything ok!')
#
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
#
# # # Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()
# #
# # # print('Eigenvalues in descending order:')
# for i in eig_pairs:
#     print(i[0])
# # # vẽ
tot = sum(eig_vals)
var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
# #
cum_var_exp = np.cumsum(var_exp)


import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

performance = var_exp[0:2]
y_pos =['PC %s' %i for i in np.arange(len(performance))]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos)
plt.ylabel('Cumulative explained variance')
plt.title('Explained variance by different principal components')
plt.show()


matrix_w = np.hstack((eig_pairs[0][1].reshape(39,1),
                      eig_pairs[1][1].reshape(39,1)))

Y = X_std.dot(matrix_w)
#
X_train, X_test, y_train, y_test = train_test_split(Y, y, test_size = 0.2, random_state = 0)
#
gnb.fit(X_train, y_train)

y_predict = gnb.predict(X_test);


target_names = ['YES', 'NO']
print(classification_report(y_test, y_predict, target_names=target_names))
# y_pred = [0, 0, 2, 2, 0, 2]

conf = confusion_matrix(y_test, y_predict, target_names);
actual_names = ['actual YES', 'actual NO']
predict_names = ['predicted YES', 'predicted NO']
print(pd.DataFrame(conf, actual_names, predict_names))