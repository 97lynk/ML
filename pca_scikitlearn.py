import numpy as np
import pandas as pd

# đọc file Wine.csv gồm 12 feature - l label
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# chuẩn hóa data: trừ toàn bộ data cho vector mean
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

# using PCA library
from sklearn.decomposition import PCA as sklearnPCA
pca = sklearnPCA(n_components = 2)
Y =  pca.fit_transform(X_std)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)


# vẽ
var_exp = [i * 100 for i in sorted(pca.explained_variance_ratio_, reverse=True)]

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
# plt.show()