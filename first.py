from sklearn import tree
#weight, 1=smooth or 0=bumpy
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [ 0, 0, 1, 1] #0=apple 1=orange 

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[130,1]]))
print(clf.predict([[180,0]]))


