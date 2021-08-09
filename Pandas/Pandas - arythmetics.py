# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:47:00 2021

@author: ogasior001
"""

import pandas as pd
import numpy as np
groceries = pd.Series(data=[3,3,1,2], index=['eggs','apples','milk','bread'])

#Arythmetics on series
print(groceries + 1)
print(groceries - 1)
print(groceries * 2)
print(groceries / 2)

#Numpy on series
print(np.sqrt(groceries))
print(np.exp(groceries))
print(np.power(groceries,2))
print(groceries['eggs'] + 1)




