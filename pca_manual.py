import numpy as np
import pandas as pd

# đọc file Wine.csv gồm 12 feature - l label
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# chuẩn hóa data: trừ toàn bộ data cho vector mean
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

cov_mat = np.cov(X_std.T)

print('Ma trận hiệp phương sai: \n%s' % cov_mat)
# eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)

cor_mat1 = np.corrcoef(X_std.T)
print('Ma trận hệ số tương quan: \n%s' % cor_mat1)

eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)

# u, s, v = np.linalg.svd(X_std.T)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()



matrix_w = np.hstack((eig_pairs[0][1].reshape(13,1),
                      eig_pairs[1][1].reshape(13,1)))

Y = X_std.dot(matrix_w)


# print('Eigenvalues in descending order:')
# for i in eig_pairs:
#     print(i[0])
# vẽ
tot = sum(eig_vals)
var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


# objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
# print(var_exp)
y_pos =['PC %s' %i for i in np.arange(len(var_exp))]
performance = var_exp

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos)
plt.ylabel('Cumulative explained variance')
plt.title('Explained variance by different principal components')
plt.show()


# print(eig_vals[0:2])
# print(var_exp[0:2])
