import numpy as np
from math import pi


b = (pi)/4
a = (pi)/4
h = 10**(-5) #when you change h closer to 0 the approximation gets better
    

def f(a,b):
    return np.sin(a+b)

def df(a,b):
    return np.array([np.cos(a+b),np.cos(a+b)])

def hf(a,b):
    return np.array([[-np.sin(a+b),-np.sin(a+b)],[-np.sin(a+b),-np.sin(a+b)]])

#Implementation of a function that approximates the gradient
def gradient(f):
    return np.array([(f(a + h, b) - f(a, b)) / h,(f(a, b + h) - f(a, b)) / h])

#Implementation of a function that approximates the hessian
def Hessian(f):
    return np.array([ [(f(a+2*h,b) - 2*f(a+h,b) + f(a,b)) / h**2,(f(a+h,b+h) - f(a+h,b) - f(a,b+h) + f(a,b)) / h**2],
                    [(f(a+h,b+h) - f(a+h,b) - f(a,b+h) + f(a,b)) / h**2,(f(a,b+2*h) - 2*f(a,b+h) + f(a,b)) / h**2]])

#approximation of gradient compared to the analytic gradient
print("error gradient", gradient(f)-df(a,b))

#approximation of hessian compared to the analytic hessian
print("error hessian", Hessian(f)-hf(a,b))


#compute the gradient and Hessian at the local extrema we found in Task 1. (function 2-1)
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


print("gradient function 2-1", df1(x,y)) # as expected since it is an extrema
print("hessian function 2-1", hf1(x,y)) # as expected since the determinant is positive and f_xx is negative which says that the extrema is a maximum



