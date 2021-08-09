# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:28:10 2021

@author: ogasior001
"""

import numpy as np

#Building array from list
my_list = [1,2,3,4,5]
X = np.array(my_list)

#Build array of zeros (shape) - it creates array of float64
X = np.zeros((3,4))
Y = np.zeros((3,4), dtype=int)

#Build array of ones (shape) - it creates array of float64
X = np.ones((3,4))
Y = np.ones((3,4), dtype=int)

#Build array of constant (shape, constant) - it creates array of constant type
X = np.full((3,4), 3)

#Build identity matrix - jedynki po przekątnej -np.eye(shape) - float64
#1000
#0100
#0010
#0001
X = np.eye(4)

#Build diagnal matrix - wartosci tylko po przekatnej
X = np.diag([10,20,30])

#Fast build one dimension array
#np.arange(stop), np.arange(startInclusive, stopExclusive), np.arange(start, stop, step)
X = np.arange(10)
X = np.arange(1, 10)
X = np.arange(1, 10, 3)

#Build 1d array with non-int step np.linspace(startInclusive, stopInclusive, n - evenly spaced numbers from start to stop)
X = np.linspace(0,25,10)
#Stop exclusive - step will change
X = np.linspace(0,25,10, endpoint=False)

#Reshape array (array, new dimension) - new dimension must fit the array size
X = np.arange(12)
X = np.reshape(X, (3,4))

#Build array with random values
X = np.random.randint(1,11, size=(5,5))

#Build array of normal distribution - np. srednia ma byc zero 
#np.random.normal(mean, standard_deviation, size)
X = np.random.normal(0, 0.1, size=(100,100))








