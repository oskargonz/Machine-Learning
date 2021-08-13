# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 12:13:36 2021

@author: ogasior001
"""

"""
I'd like to train a support vector machine, and I'd like to decide between the following parameters:
kernel: poly or rbf.
C: 0.1, 1, or 10.
"""

from sklearn.model_selection import GridSearchCV

parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}

#1 Wybieram kryterium oceny algorytmu - F1 score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
scorer = make_scorer(f1_score)

#2 Tworzę obiekt GridSearch z parametrami i wynikiem.
# Create the object.
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
# Fit the data
grid_fit = grid_obj.fit(X, y)

best_clf = grid_fit.best_estimator_