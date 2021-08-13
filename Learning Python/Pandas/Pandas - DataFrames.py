# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 18:10:29 2021

@author: ogasior001
"""

import pandas as pd

items = {'Bob' : pd.Series(data = [245, 25, 55], index = ['bike', 'pants', 'watch']),
         'Alice' : pd.Series(data = [40, 110, 500, 45], index = ['book', 'glasses', 'bike', 'pants'])}

print(items)

#Dataframes - ładna tabelka
shoppingCards = pd.DataFrame(items)
print(shoppingCards)
print(shoppingCards.index)
print("_________________________")
print(shoppingCards.columns)
print("_________________________")
print(shoppingCards.values)
print(shoppingCards.shape)

#DataFrames - specific items
bob_shopping_cards = pd.DataFrame(items, columns=['Bob'])
sel_shopping_cards = pd.DataFrame(items, index=['bike','pants'])

# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35}, 
          {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]

# We create a DataFrame 
store_items = pd.DataFrame(items2)
print()
print(store_items)

# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35}, 
          {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]

# We create a DataFrame  and provide the row index
store_items = pd.DataFrame(items2, index = ['store 1', 'store 2'])

# We display the DataFrame
print()
print(store_items)
print()

#Accessing items in DataFrame
#DataFrame[column][row]
print(store_items[['bikes']])
print(store_items.loc[['store 1']])
print(store_items['bikes']['store 2'])

#Adding columns to DataFrame
store_items['shirts'] = [15,2]
store_items['suits'] = store_items['shirts'] + store_items['pants']
store_items['new_watches'] = store_items['watches'][1:]

#Adding row to DataFrame (do Series nie mogę zrobić append new row)
new_items = [{'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4}]
new_store = pd.DataFrame(new_items, index=['store 3'])
store_items = store_items.append(new_store)

#Insert(location, label, data)
store_items.insert(3, 'shoes', [1,2,3])

#Deleting columns
store_items.pop('new_watches')
store_items = store_items.drop(['watches', 'shoes'], axis=1)

#Deleting rows
store_items = store_items.drop(['store 1', 'store 2'], axis=0)

#Rename labels / rows
store_items = store_items.rename(columns={'bikes' : 'hats'})
store_items = store_items.rename(index={'store 3' : 'last store'})

#Name indexing
store_items = store_items.set_index('pants')
print(store_items)

#Loading data from file
Google_stock = pd.read_csv('C:/DigitalAccelerator/Udacity/AI/Python programs/Pandas/GOOG.csv')
type(Google_stock)
Google_stock.shape
#print first/last 5 rows / or more/less - .head(# of rows)
Google_stock.head()
Google_stock.tail()

#Check if there are any missing data 
Google_stock.isnull().any()

#Statistics data about DataFrame
Google_stock.describe()
Google_stock['Adj Close'].describe()
print('Maximum values of each column:\n', Google_stock.max())
print('Minimum Close value:', Google_stock['Close'].min())
print('Average value of each column:\n', Google_stock.mean())
# We display the correlation between columns
Google_stock.corr()

data = pd.read_csv('C:/DigitalAccelerator/Udacity/AI/Python programs/Pandas/fake-company.csv')

#Group data by
data.groupby(['Year'])['Salary'].sum()
data.groupby(['Year', 'Department'])['Salary'].sum()
data.groupby(['Year'])['Salary'].mean()





















