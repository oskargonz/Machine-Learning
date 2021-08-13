# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 09:17:20 2021

@author: ogasior001
"""

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

#Neural Networks
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()

#Decision Trees
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()

#Support Vector Machines
from sklearn.svm import SVC
classifier = SVC()

#Using of above algorythms
data = pandas.read_csv('data.csv')
X = numpy.array(data[['x1', 'x2']])
y = numpy.array(data['y'])

classifier.fit(X,y)

