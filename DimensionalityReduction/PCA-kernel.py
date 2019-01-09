# đọc data từ file
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split


import pandas as pd
dataset = pd.read_csv('dataset_for_KernelPCA.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
col_names = dataset.columns.values[1:4].tolist()

# cột giới tính
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
X[:,1] = encoder.fit_transform(X)[:,1]


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.figure('Raw data')
ax = plt.subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], col_names, c=y)
ax.set_xlabel(col_names[0])
ax.set_ylabel(col_names[1])
ax.set_zlabel(col_names[2])
plt.show()


from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print('=======Logistic Regression without extract feature======')
print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))

kernel = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernel:
    print('=======Logistic Regression without PCA kernel=%s======' % k)
    kpca = KernelPCA(n_components=2, kernel=k)
    kpca.fit(X)
    X_new = kpca.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=0)
    clf2 = LogisticRegression(solver='lbfgs')
    clf2.fit(X_train, y_train)
    y_predict = clf2.predict(X_test)
    print(classification_report(y_test, y_predict))
    print(confusion_matrix(y_test, y_predict))

    plt.figure(k)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=y)

# eig_vecs = kpca.alphas_
# eig_vals = kpca.lambdas_
# matrix_w = np.vstack([eig_vecs[:, i] for i in range(kpca.n_components)])


