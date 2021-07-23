import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

''' SVM
The objective of the support vector machine algorithm is to find a hyperplane 
in an N-dimensional space(N — the number of features) that distinctly classifies the data points.
'''

# Read data
cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

# Set Variables
X = cancer.data
y = cancer.target

# Seperate Data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

print(x_train, y_train)
classes = ['malignant' 'benign']

# Train model
clf = KNeighborsClassifier(n_neighbors=9)
# clf = svm.SVC(kernel="linear", C=1)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test,y_pred)

print(acc)
