# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:16:39 2021

@author: ogasior001
"""

cast = ["Barney Stinson", "Robin Scherbatsky", "Ted Mosby", "Lily Aldrin", "Marshall Eriksen"]
heights = [72, 68, 72, 66, 76]

# write your for loop here
#for i, character in enumerate(cast):
 #   cast[i] = character + " " + str(heights[i])

for i in range(len(cast)):
    cast[i] = cast[i] + " " + str(heights[i])
    

print(cast)

