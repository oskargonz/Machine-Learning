# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:39:18 2021

@author: ogasior001
"""

import pandas as pd

groceries = pd.Series(data=[3,6,'Yes', 'No'], index=['eggs','apples','milk','bread'])
print(groceries)
print(groceries.shape)
print(groceries.ndim)
print(groceries.size)
print(groceries.index)
print(groceries.values)
print('banana' in groceries)
print()

#Accessing data
print('XXXXXXXXXXX')
print(groceries['eggs'])
print()
print(groceries[['eggs', 'milk']])
print()
print(groceries[0])
print(groceries[[1,2]])
print()

#Deleting
#Without changing original
print(groceries.drop('eggs'))
print()
print(groceries)
print()
#With changing original
print(groceries.drop('eggs', inplace=True))
print()
print(groceries)




