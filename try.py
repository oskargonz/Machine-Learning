# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:53:45 2021

@author: ogasior001
"""

try:
    x = 9 / "s"
except Exception as e:
   print("Exception occurred: {}".format(e))

try:
    x = 2/0
except ZeroDivisionError as e:
   print("ZeroDivisionError occurred: {}".format(e))

while True:
    try:
        x = int(input("Type int value: "))
        break
    except:
        print("That is not an int value")
    finally:
        print("This is finally statement printed in both cases - exception and correct valued")
        
while True:
    try:
        x = int(input("Type int value: "))
        break
    except ValueError:
        print("Now I can interrupt the loop by ctrl + c. Previously except handeled also 'KeyboardInterrupt'")
        
