import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# đọc file dataset_for_PCA_LDA.csv gồm 39 feature - l label
dataset = pd.read_csv('dataset_for_PCA_LDA.csv')
X = dataset.iloc[:, 0:39].values
y = dataset.iloc[:, 39].values

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

### Code phần 2.1: các nhóm code LDA
lda = LDA()
X_sklearn =  lda.fit_transform(X, y)

### Code phần 2.2: dùng chính giải thuật A ở trên để làm bài toán classification dựa trên dataset với các LD mới này
X_train, X_test, y_train, y_test = train_test_split(X_sklearn, y, test_size=0.2, random_state=0)
clf_2 = GaussianNB()
clf_2.fit(X_train, y_train)
y_predict = clf_2.predict(X_test)

conf_2 = confusion_matrix(y_test, y_predict, target_names)
print('Using Navie Bayes after apply LDA ')
print('===================Confusion Matrix 2==================\n%s' % pd.DataFrame(conf_2, actual_names, predict_names))
print('===================Report 2============================\n%s' % classification_report(y_test, y_predict, target_names))


# TODO chưa xong LDA
