# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:46:12 2021

@author: ogasior001
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Load the Census dataset
data = pd.read_csv('C:/DigitalAccelerator/Udacity/AI/Repo/Machine-Learning/Learning Machine learning/Project 3/finding_donors/census.csv')

# Success - Display the first record
display(data.head(n=1))

df = pd.DataFrame(data)

#Total number of records
n_records = df.index

#Number of records where individual's income is more than $50,000
n_greater_50k = df.loc[df['income'] == '>50K']

#Number of records where individual's income is at most $50,000
n_at_most_50k = df.loc[df['income'] == '<=50K']

#Percentage of individuals whose income is more than $50,000
greater_percent = (len(n_greater_50k.index) / len(n_records)) * 100

#Print the results
print("Total number of records: {}".format(len(n_records)))
print("Individuals making more than $50,000: {}".format(len(n_greater_50k.index)))
print("Individuals making at most $50,000: {}".format(len(n_at_most_50k.index)))
print("Percentage of individuals making more than $50,000: {}%".format(round(greater_percent, 2)))

#CODE FROM UDACITY
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)
# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))
#END OF CODE FROM UDACITY

#ONE-HOT ENCODING

features_final = pd.get_dummies(features_log_minmax_transform)
income = np.where(income_raw.str.contains(">50K"), 1, 0)

#Number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
# print(encoded)

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

#Naive Predictor Performace - sprawdzam jakie beda wyniki jesli oznacze wszystkich jako True (1)
TP = np.sum(income)
FN = len(income) - TP
TN = 0
FP = 0

#Calculate accuracy, precision and recall
accuracy = (TP + TN) / len(income)
recall = TP / (TP + FN)
precision = TP / (TP + FP)

#Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta = 0.5
fscore = (1 + np.power(beta, 2)) * ((precision * recall) / ((np.power(beta, 2) * precision) + recall))

# Print the results 
print("\nManual calculations: ")
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
print("Precision: " + str(precision))
print("Recall: " + str(recall))

print("\nAuto calculations: ")
acc = accuracy_score(income, np.ones(45222))
fbeta = fbeta_score(income, np.ones(45222), beta = 0.5)
prec = precision_score(income, np.ones(45222))
rec = recall_score(income, np.ones(45222))

print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(acc, fbeta))
print("Precision: " + str(prec))
print("Recall: " + str(rec))






















