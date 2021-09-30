# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 16:16:47 2021

@author: ogasior001
"""

#program normalizuje dane. Dzieki temu mozna uzywac kilku paramertow np. waga + wzrost i zaden parametr nie jest 'faworyzowany'

from sklearn.preprocessing import MinMaxScaler
import numpy as np

weights = np.array([[115.],[140.],[175.]])
scaler = MinMaxScaler()
rescaled_weights = scaler.fit_transform(weights)
print(rescaled_weights)