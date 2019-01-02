import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

X = np.array([[-1, -1, 1], [-2, -1, 5], [-3, -2, 3], [1, 1, 1], [2, 1, 1], [3, 2, 1]])
y = np.array([1, 1, 1, 4, 2, 3])
clf = LDA()
clf.fit(X, y)
print(clf.predict([[-0.8, -1, 0]]))