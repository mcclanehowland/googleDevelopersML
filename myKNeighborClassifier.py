import random
import math
import numpy as np

class ScrappyKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def calc_distance(self, x1, x2): #len(x1) = len(x2)
        differences = np.subtract(x1, x2)
        square = np.square(differences)
        distance_proxy = np.sum(square)
        return distance_proxy

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            best_label = -1
            min_distance = self.calc_distance(self.x_train[0], x_test[0])
            for i in range(len(self.x_train)):
                d = self.calc_distance(self.x_train[i], row)
                if(d < min_distance):
                    min_distance = d
                    best_label = self.y_train[i]
            predictions.append(best_label)

        return predictions

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)


classifier = ScrappyKNN()

classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

