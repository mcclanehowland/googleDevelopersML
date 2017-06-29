import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
import csv

iris = load_iris()
#get the training data

#remove the test data from the training data
testing_indices = [0, 50, 100] #one of each flower
training_data = np.delete(iris.data, testing_indices, axis=0)
training_labels = np.delete(iris.target, testing_indices, axis=0)
#get the test data
testing_data = iris.data[testing_indices]
testing_labels = iris.target[testing_indices]

#train the classifier with the training data
clf = tree.DecisionTreeClassifier()
clf.fit(training_data, training_labels)

#predict based on testing data
predictions = clf.predict(testing_data)

#check if those predictions were right
for i in range(len(predictions)):
    if(predictions[i] == testing_labels[i]):
        print("Success!")
    else:
        print("Failure!")






