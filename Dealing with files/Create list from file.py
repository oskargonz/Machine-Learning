# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:20:58 2021

@author: ogasior001
"""

def create_cast_list(filename):
    cast_list = []
    #use with to open the file filename
    with open("C:/DigitalAccelerator/Udacity/AI/Python programs/Dealing with files/flying_circus_cast.txt") as f:
        for line in f:
            file_data = line.split(',')
            cast_list.append(file_data[0])
    #use the for loop syntax to process each line
    #and add the actor name to cast_list

    return cast_list

cast_list = create_cast_list('C:/DigitalAccelerator/Udacity/AI/Python programs/Dealing with files/flying_circus_cast.txt')
for actor in cast_list:
    print(actor)