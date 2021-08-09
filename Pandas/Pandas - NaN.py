# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:51:02 2021

@author: ogasior001
"""

import pandas as pd

items2 = [{'bikes': 20, 'pants': 30, 'watches': 35, 'shirts': 15, 'shoes':8, 'suits':45},
{'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5, 'shirts': 2, 'shoes':5, 'suits':7},
{'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4, 'shoes':10}]

store_items = pd.DataFrame(items2, index = ['store 1', 'store 2', 'store 3'])

#I can count # of NaN values as .isnull daje True = 1 i False = 0
x = store_items.isnull()
#Number of NaN in each column
x = store_items.isnull().sum()
#Total # of NaN values
x = store_items.isnull().sum().sum()

#Counting of non-NaN values
#Number of non-NaN in each column
x = store_items.count()
#Total # of non-NaN values
x = store_items.count().sum()

#Drop NaN values without modification of original data
#Eliminate any row with NaN values
store_items.dropna(axis=0)
#Eliminate any column with NaN values
store_items.dropna(axis=1)
#with modification of original data
store_items.dropna(axis=1, inplace=True)

items2 = [{'bikes': 20, 'pants': 30, 'watches': 35, 'shirts': 15, 'shoes':8, 'suits':45},
{'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5, 'shirts': 2, 'shoes':5, 'suits':7},
{'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4, 'shoes':10}]
store_items = pd.DataFrame(items2, index = ['store 1', 'store 2', 'store 3'])

print(store_items)
#Replacing NaN with values
store_items = store_items.fillna(0)
#Uzulelnienie NaN poprzednimi wartosciami z kolumny
store_items = store_items.fillna(method = 'ffill', axis = 0)
#Uzulelnienie NaN poprzednimi wartosciami z wiersza
store_items = store_items.fillna(method = 'ffill', axis = 1)
#Uzulelnienie NaN kolejnymi wartosciami z kolumny
store_items = store_items.fillna(method = 'backfill', axis = 0)
#Uzulelnienie NaN kolejnymi wartosciami z wiersza
store_items = store_items.fillna(method = 'backfill', axis = 1)

# We replace NaN values by using linear interpolation using column values
store_items.interpolate(method = 'linear', axis = 0)

# We replace NaN values by using linear interpolation using row values
store_items.interpolate(method = 'linear', axis = 1)





