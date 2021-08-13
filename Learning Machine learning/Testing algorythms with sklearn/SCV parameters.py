# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:18:54 2021

@author: ogasior001
"""

#Support Vector Machines
from sklearn.svm import SVC
classifier = SVC()
#SVC parameters:
#kernel (string): 'linear', 'poly', 'rbf'.
#degree (integer): This is the degree of the polynomial kernel, if that's the kernel you picked (goes with poly kernel).
#gamma (float): The gamma parameter (goes with rbf kernel).
#C (float): The C parameter.

#e.g. classifier = SVC(kernel = None, degree = None, gamma = None, C = None)