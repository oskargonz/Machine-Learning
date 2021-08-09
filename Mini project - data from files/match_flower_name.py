# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 16:29:53 2021

@author: ogasior001
"""


# HINT: create a dictionary from flowers.txt
flowers = {}
with open ("C:/DigitalAccelerator/Udacity/AI/Python programs/Mini project - data from files/flowers.txt") as f:
    for line in f:
        key, value = line.split(": ")
        flowers[key] = value.strip()
# HINT: create a function to ask for user's first and last 

try:
    name = str(input("Enter your First [space] Last name only:"))
    print("Unique flower name with the first letter: " + flowers[name[0]])
except ValueError:
    print("Wrong data type. Insert your name again")

# print the desired output
