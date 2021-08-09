# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 09:40:34 2021

@author: ogasior001
"""

f = open('C:/DigitalAccelerator/Udacity/AI/Python programs/Dealing with files/some_file.txt', 'w')
#file_data = f.read()
f.write("Some new text")
f.close()

with open("C:/DigitalAccelerator/Udacity/AI/Python programs/Dealing with files/camelot.txt") as song:
    print(song.read(2))
    print(song.read(8))
    print(song.read())

#Better way to deal with files
with open('C:/DigitalAccelerator/Udacity/AI/Python programs/Dealing with files/some_file.txt', 'r') as f:
    file_data = f.read()

#OSError example - too many open files
"""
files = []
for i in range(10000):
    files.append(open('some_file.txt', 'r'))
    print(i)
    """