# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 08:25:49 2021

@author: ogasior001
"""
import numpy as np

#Deleting elements from array
#np.delete(array, [number of element])
X = np.array([1,2,3,4,5])
X = np.delete(X, [2])
#Multi-dimension arrays - np.delete(array, element, axis=0/1) 0 - row, 1 - column
Y = np.arange(1,10).reshape(3,3)
Y = np.delete(Y, 0, axis=0)

#Appending elements, axis=0 
X = np.array([1,2,3,4,5])
X = np.append(X, 6)
#Multi-dimension array - np.append(array, [element], axis=0/1)
Y = np.arange(1,10).reshape(3,3)
Y = np.append(Y, [[1,2,3]], axis=0)

#Insert values - eg. insert row between row 1 and 2
X = np.array([[1,2,3],[4,5,6]])
X = np.insert(X, 1, [0,0,0], axis=0)

#Stacking arrays
X = np.array([1,2])
Y = np.array([[3,4], [5,6]])
Z = np.vstack((X, Y))
W = np.hstack((Y, X.reshape(2,1)))

#Slicing arrays
X = np.arange(1,21).reshape(4,5)
Z = X[1:4, 2:5]

#Copy
X = np.arange(1,21).reshape(4,5)
Z = np.copy(X[1:, 2:])
z = X[1:, 2:].copy()

#Tworzenie 1d array z przekątnej 2d array. diag(array, k=1/2/3/-1/-2/...) k - przekątna powyżej lub poniżej głównej przekątnej
X = np.arange(1,21).reshape(4,5)
Z = np.diag(X, k=1)
print(X)
print()
print(Z)

