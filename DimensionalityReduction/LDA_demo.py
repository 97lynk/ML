import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# đọc file dataset_for_PCA_LDA.csv gồm 39 feature - l label
dataset = pd.read_csv('dataset_for_PCA_LDA.csv')
X = dataset.iloc[:, 0:39].values
y = dataset.iloc[:, 39].values

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


from sklearn.preprocessing import LabelEncoder

lb_enc = LabelEncoder()
lb_enc.fit(y)
y = lb_enc.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

target_names = ['%s' % i for i in lb_enc.classes_]
actual_names = ['actual %s' % i for i in lb_enc.classes_]
predict_names = ['predicted %s' % i for i in lb_enc.classes_]
y_test = lb_enc.inverse_transform(y_test)
y_predict = lb_enc.inverse_transform(y_predict)

conf_1 = confusion_matrix(y_test, y_predict)
print('Using Navie Bayes before apply PCA')
print('===================Confusion Matrix 1==================\n%s' % pd.DataFrame(conf_1, actual_names, predict_names))
print('===================Report 1============================\n%s' % classification_report(y_test, y_predict,
                                                                                            target_names))
lda = LDA(n_components=1)
X_sklearn = lda.fit_transform(X, y)

### Code phần 2.2: dùng chính giải thuật A ở trên để làm bài toán classification dựa trên dataset với các LD mới này
X_train, X_test, y_train, y_test = train_test_split(X_sklearn, y, test_size=0.2, random_state=0)
clf_2 = LogisticRegression(random_state=0)
clf_2.fit(X_train, y_train)
y_predict = clf_2.predict(X_test)

y_test = lb_enc.inverse_transform(y_test)
y_predict = lb_enc.inverse_transform(y_predict)
conf_2 = confusion_matrix(y_test, y_predict, target_names)
print('Using Navie Bayes after apply LDA ')
print('===================Confusion Matrix 2==================\n%s' % pd.DataFrame(conf_2, actual_names, predict_names))
print('===================Report 2============================\n%s' % classification_report(y_test, y_predict,
                                                                                            target_names))

# TODO chưa xong LDA