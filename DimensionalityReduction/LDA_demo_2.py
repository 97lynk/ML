# đọc data từ file
import pandas as pd

from DimensionalityReduction.MyLDA import LinearDiscriminantAnalysis

from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
import pandas as pd

from DimensionalityReduction.PCA import colors

dataset = pd.read_csv('dataset_for_PCA_LDA.csv')
X = dataset.iloc[:, 0:39].values
y = dataset.iloc[:, 39].values
#
# dataset = pd.read_csv('dataset_for_PCA_LDA.csv')
# X = dataset.iloc[:, 0:39].values
# y = dataset.iloc[:, 39].values

# chuẩn hóa data về 0 -> 1
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
# encode label NO/YES -> 0/1
from sklearn.preprocessing import LabelEncoder

lb_enc = LabelEncoder()
y = lb_enc.fit_transform(y)

# split data 20% cho testing
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=0)
# clf = LogisticRegression(solver='lbfgs')
# clf.fit(X_train, y_train)
# y_predict = clf.predict(X_test)
#
# # in report
# target_names = ['YES', 'NO']
# actual_names = ['actual YES', 'actual NO']
# predict_names = ['predicted YES', 'predicted NO']
# y_test_report, y_pred_report = lb_enc.inverse_transform(y_test), lb_enc.inverse_transform(y_predict)
#
# from sklearn.metrics import confusion_matrix, classification_report
#
# conf_1 = confusion_matrix(y_test_report, y_pred_report, target_names)
# table_conf_1 = pd.DataFrame(conf_1, actual_names, predict_names)
# report_1 = classification_report(y_test_report, y_pred_report, target_names)
# print('Using Logistic Regression after apply LDA')
# print('===================Confusion Matrix 1==================\n%s' % table_conf_1)
# print('===================Report 1============================\n%s' % report_1)

### LDA manual
import numpy as np

# n_classes = 2
# n_features = X_m.shape[1]
#
# mean_vectors = []
# for cl in range(n_classes):
#     mean_vectors.append(np.mean(X_m[y == cl], axis=0))
#
# # calculate Scatter Within matrix
# S_W = np.zeros((n_features, n_features))
# for cl, mv in zip(range(n_classes), mean_vectors):
#     class_sc_mat = np.zeros((n_features, n_features))
#     for row in X_m[y == cl]:
#         row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1)
#         class_sc_mat += (row - mv).dot((row - mv).T)
#     S_W += class_sc_mat
#
# # calculate Scatter Between matrix
# overall_mean = np.mean(X_std, axis=0)
# S_B = np.zeros((n_features, n_features))
# for i, mean_vec in enumerate(mean_vectors):
#     n = X_m[y == i + 1, :].shape[0]
#     mean_vec = mean_vec.reshape(n_features, 1)
#     overall_mean = overall_mean.reshape(n_features, 1)
#     S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
#
# # decomposite eigens
# eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# sort_idx = np.argsort(eig_vals)[::-1]
# eig_vals, eig_vecs = eig_vals[sort_idx].real, eig_vecs[:, sort_idx].real
#
# # make matrix_w
# n_new = 1
# matrix_w = np.vstack([eig_vecs[:, i] for i in range(n_new)]).T
# x_new = X_m.dot(matrix_w)
X_1, X_2 = X, X
lda1 = LinearDiscriminantAnalysis(n_discriminants=1)
lda1.fit(X_1, y)
X_new_1 = lda1.transform(X_1)
tot = sum(lda1.e_vals_)
var_exp = [(i / tot)*100 for i in sorted(lda1.e_vals_, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda2 = LDA(n_components=1)
X_new_2 = lda2.fit_transform(X_2, y)
import matplotlib.pyplot as plt


# plt.figure(1)
# plt.plot(X_new_1[:, 0].real, np.zeros_like(X_new_1[:, 0]), colors=y)
# plt.show()

# plt.figure(2)
# plt.plot(X_new_2[:, 0], np.zeros_like(X_new_2), c=y)
# plt.show()