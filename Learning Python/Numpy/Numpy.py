# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:26:07 2021

@author: ogasior001
"""

import numpy as np

X = np.arange(25).reshape(5,5)
print(X)

print("\nThis is X[X>10]:\n", X[X>10])

print("\nThis is X [X <= 7]:\n",  X [X <= 7])

X[(X > 10) & (X < 17)] = -1

print("\nThis is X[(X > 10) & (X < 17)] = -1:\n", X)

x = np.array([1,2,3,4,5])
y = np.array([1,7,2,8,4])

# We use set operations to compare x and y:
print()
print('The elements that are both in x and y:', np.intersect1d(x,y))
print('The elements that are in x that are not in y:', np.setdiff1d(x,y))
print('All the elements of x and y:',np.union1d(x,y))
print("\nSorting example:")
print("y =", y)
print("np.sort(y): ", np.sort(y))  
print("y after np.sort(y)", y)
y.sort()
print("y after y.sort()", y)

print()
X = np.random.randint(1,11, size=(5,5))
print("\nThis is random 2 dimensions array:\n",X)

print("\nRandom array sorted by rows:\n", np.sort(X, axis=0))
print("\nRandom array sorted by columns:\n", np.sort(X, axis=1))

print("\nUnique values of multi dimensional arrays\n", np.unique(X))