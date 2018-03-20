# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 11:03:21 2018

@author: duzhe
"""

from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
#features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
labels = [0, 0, 1, 1]
#labels = ["apple", "apple", "orange", "orange"]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[150, 0]]))