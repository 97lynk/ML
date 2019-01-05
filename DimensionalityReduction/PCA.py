import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# đọc file dataset_for_PCA_LDA.csv gồm 39 feature - l label
dataset = pd.read_csv('dataset_for_PCA_LDA.csv')
X = dataset.iloc[:, 0:39].values
y = dataset.iloc[:, 39].values
# n = 3  # n_neighbors
n_pc = 3 # number of principal component

### Code phần 1: dùng bất kì giải thuật classification nào mà sinh viên đã học để phân loại
# Giải thuật Navie Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# clf = KNeighborsClassifier(n_neighbors=n)
clf = GaussianNB()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
# Plot kết quả dùng giải thuật Navie Bayes trong đồ thị 2D, 3D tùy ý


# Kết quả classification dùng giải thuật Navie Bayes ra được Confusion Matrix 1
target_names = ['YES', 'NO']

actual_names = ['actual YES', 'actual NO']
predict_names = ['predicted YES', 'predicted NO']
conf_1 = confusion_matrix(y_test, y_predict, target_names);
print('Using Navie Bayes before apply PCA')
print('===================Confusion Matrix 1==================\n%s' % pd.DataFrame(conf_1, actual_names, predict_names))
print('===================Report 1============================\n%s' % classification_report(y_test, y_predict,
                                                                                            target_names))

print('\n')

### Code phần 2.1: các nhóm tùy ý code PCA
## tính thủ công cov matrix, cor matrix và trị riêng vextor riêng
# chuẩn hóa data
X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=n_pc)
X_sklearn = pca.fit_transform(X_std)

cov_mat = np.cov(X_std.T)
print('Covariance matrix: \n%s' % cov_mat)
cor_mat = np.corrcoef(X_std.T)
print('Correlation matrix: \n%s' % cor_mat)
eig_vals, eig_vecs = np.linalg.eig(cor_mat)
# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)
# gộp trị riêng vector riêng lại thành từng cặp
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
# xắp xếp các cặp từ cao -> thấp theo trị riêng
eig_pairs.sort()
eig_pairs.reverse()
#in xem thử
print('Eigenvalues in descending order:')
for value in eig_pairs:
    print(value[0])

### Code phần 2.2: dùng chính giải thuật Navie Bayes ở trên để làm bài toán classification dựa trên dataset với các PC mới này
X_train, X_test, y_train, y_test = train_test_split(X_sklearn, y, test_size=0.2, random_state=0)
# clf_2 = KNeighborsClassifier(n_neighbors=n)
clf_2 = GaussianNB()
clf_2.fit(X_train, y_train)
y_predict = clf_2.predict(X_test)

conf_2 = confusion_matrix(y_test, y_predict, target_names)
print('Using Navie Bayes after apply PCA with PC = %d' % (n_pc))
print('===================Confusion Matrix 2==================\n%s' % pd.DataFrame(conf_2, actual_names, predict_names))
print('===================Report 2============================\n%s' % classification_report(y_test, y_predict,
                                                                                            target_names))
# tính phần trăm explained variance
var_exp = [i * 100 for i in sorted(pca.explained_variance_ratio_, reverse=True)]
print('The first %d(s) principal components contain %.2f%% of the information' % (n_pc, sum(var_exp)))
# cộng dồn
cum_var_exp = np.cumsum(var_exp)

plt.rcdefaults()
# tạo label cho từng bar với PC
x_pos = ['PC %s' % i for i in np.arange(len(var_exp))]

for index, pc in enumerate(x_pos):
    print('      %s = %.2f%%' %(pc, var_exp[index]))

plt.figure(figsize=(6, 4))
# vẽ chấm cộng dồn qua từng PC
plt.plot(cum_var_exp, color='green')
# nối các chấm trên
plt.scatter(x_pos, cum_var_exp, color='orange', label='cumulative explained variance')
# vẽ các bar với giá trị là explained variance
bars = plt.bar(x_pos, var_exp, align='center', alpha=0.5, label='individual explained variance')

# thêm text cho bar
for index, bar in enumerate(bars):
    plt.text(bar.get_x() + .2, bar.get_height() - 5, '{:.2f}%'.format(var_exp[index]) , fontsize=18, color='blue')

# thêm text cho chấm
for i, value in enumerate(cum_var_exp):
    plt.text(x_pos[i], cum_var_exp[i] + 2, '{:.2f}%'.format(value))

plt.xticks(x_pos)
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.title('Explained variance by different principal components')
plt.legend(loc='best')
plt.tight_layout()

if n_pc == 2:
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('YES', 'NO'),
                        ('green','red')):
        plt.scatter(X_sklearn[y==lab, 0],
                    X_sklearn[y==lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()

if n_pc == 3:
    plt.figure(2)
    colors = {
        'YES': 'green',
        'NO': 'red'
    }
    ax = plt.subplot(111, projection='3d')
    for index, val in enumerate(X_sklearn):
        xdata = val[0]
        ydata = val[1]
        zdata = val[2]
        ax.scatter(xdata, ydata, zdata, color=colors.get(y[index]))
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

plt.show()

print('===============Conclusion==================')
print('Do PCA giảm chiều dữ liệu từ 39 features xuống còn %s features và chỉ còn %.2f%% '
      'thông tin nên precision và recall giảm' %(n_pc, sum(var_exp)))