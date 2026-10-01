# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 22:00:19 2026

@author: Bruno
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi


b = (pi)/4
a = (pi)/4
h = 10**(-5)
    

def f(a,b):
    return np.sin(a+b)

def df(a,b):
    return np.array([np.cos(a+b),np.cos(a+b)])

def hf(a,b):
    return np.array([[-np.sin(a+b),-np.sin(a+b)],[-np.sin(a+b),-np.sin(a+b)]])

def gradient(f):
    return np.array([(f(a + h, b) - f(a, b)) / h,(f(a, b + h) - f(a, b)) / h])


def Hessian(f):
    return np.array([ [(f(a+2*h,b) - 2*f(a+h,b) + f(a,b)) / h**2,(f(a+h,b+h) - f(a+h,b) - f(a,b+h) + f(a,b)) / h**2],
                    [(f(a+h,b+h) - f(a+h,b) - f(a,b+h) + f(a,b)) / h**2,(f(a,b+2*h) - 2*f(a,b+h) + f(a,b)) / h**2]])

print(gradient(f))
print("error", gradient(f)-df(a,b))
print("error", Hessian(f)-hf(a,b))



x = 1
y = 2
def f1(x,y):
    return 8*x*y-4*(x**2)*y-2*x*y**2+(x**2)*y**2

def df1(x,y):
    fx = 8*y-8*x*y-2*y**2+2*x*y**2
    fy = 8*x-4*x**2-4*x*y+2*(x**2)*y

    return np.array([fx, fy])

def hf1(x,y):
    fxx = -8*y+2*y**2
    fxy = 8-8*x-4*y+4*x*y
    fyy = -4*x+2*x**2

    return np.array([[fxx, fxy],[fxy, fyy]])


print(df1(x,y))
print(hf1(x,y))




