# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:39:12 2021

@author: ogasior001
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit


# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('C:/DigitalAccelerator/Udacity/AI/Repo/Machine-Learning/Learning Machine learning/Project 1/boston_housing/housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

# TODO: Minimum price of the data
minimum_price = prices.min(axis=0)

# TODO: Maximum price of the data
maximum_price = prices.max(axis=0)

# TODO: Mean price of the data
mean_price = prices.mean()

# TODO: Median price of the data
median_price = prices.median(axis=0)

# TODO: Standard deviation of prices of the data
std_price = prices.std(axis=0)

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))

#Question 1
"""
Would you expect a home that has an 'RM' value(number of rooms) of 6 be worth more or less than a home that has an 'RM' value of 7?
Would you expect a neighborhood that has an 'LSTAT' value (percent of lower income level households) of 15 have home prices be worth more or less than a neighborhood that has an 'LSTAT' value of 20?
Would you expect a neighborhood that has an 'PTRATIO' value(ratio of students to teachers) of 10 have home prices be worth more or less than a neighborhood that has an 'PTRATIO' value of 15?

**Answer: 
    To answer this question I draw 3 scatter plots:
plt.scatter(data['RM'], data['MEDV'])
plt.scatter(data['LSTAT'], data['MEDV'])
plt.scatter(data['PTRATIO'], data['MEDV'])

    
The higher 'RM' value, the higher home value, 
the higher 'LSTAT' value, the lower home value
'PTRATIO' chart is unstructured.
"""


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score

# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("\nModel has a coefficient of determination, R^2, of {:.3f}.".format(score))

#Question 2 - Goodness of Fit
"""
Would you consider this model to have successfully captured the variation of the target variable?
Why or why not?

Answer:
    Yes, I would the model to have successfully captured the variation of the target variable as model has a coefficient of determination, R^2, of 0.923.
"""


RM_train, RM_test, yRM_train, yRM_test = train_test_split(features['RM'], prices, test_size=0.2, random_state=42)
LSTAT_train, LSTAT_test, yLSTAT_train, yLSTAT_test = train_test_split(features['LSTAT'], prices, test_size=0.2, random_state=42)
PTRATIO_train, PTRATIO_test, yPTRATIO_train, yPTRATIO_test = train_test_split(features['PTRATIO'], prices, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)

#Question 3
"""
What is the benefit to splitting a dataset into some ratio of training and testing subsets for a learning algorithm?

Answer: I think that low testing size may cause overfitting, when high testing size mat cause underfitting.
"""

vs.ModelLearning(features, prices)

#Question 4 - Learning the Data
"""
Choose one of the graphs above and state the maximum depth for the model.
What happens to the score of the training curve as more training points are added? What about the testing curve?
Would having more training points benefit the model?

Ansewer: I think that the best is max_depth = 3 as both training and testing curves are close to each other and have relatively high score.
Above 300 testing points the score is constant thus I assume that more testing points are unnecessary.
Pros of adding more testing points is that the model would be a little better.
Cons of adding more testing points is that we will have less data for testing.
"""

vs.ModelComplexity(X_train, y_train)

#Question 5 - Bias-Variance Tradeoff
"""
When the model is trained with a maximum depth of 1, does the model suffer from high bias or from high variance?
How about when the model is trained with a maximum depth of 10? What visual cues in the graph justify your conclusions?

Answer: 
    When the model is trained with a maximum depth of 1, the model suffer from high bias (high underfitting). There is a low score in the graph.
    When the model is trained with a maximum depth of 10, the model suffer from high variance (high overfitting). There is a big gap between training and validation score.
"""

#Question 6 - Best-Guess Optimal Model
"""
Which maximum depth do you think results in a model that best generalizes to unseen data?
What intuition lead you to this answer?

Answer:     
    I think that the best would be a maximum depth of 3.
    At maximum depth of 3 testing and validation score are high and from maximum depth of 4 the gap between curves grows.
"""

#Question 7 - Grid Search
"""
What is the grid search technique?
How it can be applied to optimize a learning algorithm?

Answer: 
    Grid search is a technique used to find the best parameters e.g. maximum depth for the model. 
    I found is the most useful for SVC algotythm where I had to use multiple parameters.
    It assess parameters based on score e.g. F1score.
"""

#Question 8 - Cross-Validation
"""
What is the k-fold cross-validation training technique?
What benefit does this technique provide for grid search when optimizing a model?

Answer: 
    k-fold cross-validation is a solution for not to through testing data from the training model. We need to break data into k buckets.
    The we train the model k-times, each time using different bucket as a testing set. Then average the result to get final model.
    This technique gives better accuracy and less randomize F1score to assess the model.
"""


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(n_splits=10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)
    print(scoring_fnc)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    #grid = GridSearchCV(regressor, params, scoring_fnc, cv_sets)
    grid = GridSearchCV(regressor,params,scoring_fnc,cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

#Question 9 - Optimal Model
"""
What maximum depth does the optimal model have? How does this result compare to your guess in Question 6?

Answer: 
    Parameter 'max_depth' is 4 for the optimal model.
    In compare to Question 6 the result is different, as there was 'max_depth' equal to 3.
"""

# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))

#Question 10 - Predicting Selling Prices
"""
What price would you recommend each client sell his/her home at?
Do these prices seem reasonable given the values for the respective features?

Answer:
    It seems reasonable:
    Predicted selling price for Client 1's home: $403,025.00
    Predicted selling price for Client 2's home: $237,478.72
    Predicted selling price for Client 3's home: $931,636.36
"""


vs.PredictTrials(features, prices, fit_model, client_data)

#Question 11 - Applicability
"""
In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting.

Answer:
    I think that this model can perform some predictions, however it does not include many factors that should.
    There are many other parameters that have an influence on the price e.g. home standard, area (not only number of rooms), old/new house, and many more.
    Moreover data used for training this model is too old and does not include specific cities.
    And finally I think that more important is home itself than neighbourhood.
    Thus I think that this model should not be used in a real-world setting.
"""















