# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:25:37 2021

@author: ogasior001
"""

def generate_password():
    return random.choice(word_list) + random.choice(word_list) + random.choice(word_list)

def generate_password():
    return ''.join(random.sample(word_list,3))