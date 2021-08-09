# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:07:18 2021

@author: ogasior001
"""

names = input("Enter names separated by commas: ").title().split(",")
assignments = input("Enter assignment counts separated by commas: ").split(",")
grades = input("Enter grades separated by commas: ").split(",")

for i in range(len(names)):
    message = "Hi {},\n\nThis is a reminder that you have {} assignments left to \
submit before you can graduate. You're current grade is {} and can increase \
to 5 if you submit all assignments before the due date.\n\n".format(names[i-1], assignments[i-1], grades[i-1])
    print(message)

# write a for loop that iterates through each set of names, assignments, and grades to print each student's message
