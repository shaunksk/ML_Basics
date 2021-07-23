import tensorflow
import tensorflow.keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read Data taken from https://archive.ics.uci.edu/ml/datasets/Student+Performance
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

# Set Y and X variables
predict = "G3"                                  # G3 is the final grade
X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

# Separate Data into testing and training
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Train model
'''
best = 0
for _ in range(30):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # Pick model, fit model and assess accuracy
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: ", acc)

    # save model if model accuracy is the best
    if acc > best:
        print("best")
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

# Model Predictions
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Plot graph
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()