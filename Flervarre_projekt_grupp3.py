# -*- coding: utf-8 -*-
"""
MATB21 Programming Project, Programming Project 3
Folke Adolfsson, Bruno Benyamine Remahl, Arvid Brenner, Sixten Midsem

Task 1: Folke, Task 2: Bruno, Task 3 and 4: Arvid
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.optimize import fsolve

# All figures are saved as pdf in the subfolder figures, next to this script
script_folder = os.path.dirname(os.path.abspath(__file__))
figure_folder = os.path.join(script_folder, "figures")
os.makedirs(figure_folder, exist_ok=True)


def save_figure(fig, name):
    fig.savefig(os.path.join(figure_folder, name + ".pdf"), bbox_inches="tight")

# Interactive mode: plt.show() in Task 1 does not stop the script, so all four
# tasks run directly. plt.show(block=True) at the end keeps all figures open.
plt.ion()

# Limit 1 is 0/0 on the line y = x, numpy gives nan there and those points
# are left empty in the plots, so we turn off the warnings for this
np.seterr(divide='ignore', invalid='ignore')

# Print every line directly, also when the output is not shown in a terminal
sys.stdout.reconfigure(line_buffering=True)


def print_header(text):
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)


# =============================================================================
# Task 1
# =============================================================================

print_header("Task 1: Limits and local extrema")
print("Only figures in this task, the answers are in the comments")

# Task 1

# All limits and functions are plotted, each in its own figure
# This was done to make it easier to read the result from the question asked

# This plotting is universal for all limits and functions in task 1
def plot_task1(name):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fig2, ax2 = plt.subplots()
    ax.plot_surface(X,Y,z,alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(zlabel)
    cs = ax2.contourf(X,Y,z,n)
    ax2.set_title(contlabel)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    cbar = fig2.colorbar(cs)
    cs2 = ax2.contour(cs, levels = cs.levels[::2], colors='r')
    cbar.add_lines(cs2)
    cbar.ax.set_ylabel(zlabel)
    save_figure(fig, "task1_" + name + "_surface")
    save_figure(fig2, "task1_" + name + "_contour")

# All of the limits in (1)

# Limit 1, the limit is 1/2
x = y = np.linspace(0, 2, 1000)
X, Y = np.meshgrid(x,y)
z = (X**2-X*Y)/(X**2-Y**2)
n = 10
title = "Surface around limit 1"
zlabel = "z"
contlabel = "Contour around limit 1"
plot_task1("limit1")

# Limit 2, the limit does not exist
x = y = np.linspace(-1, 1, 1000)
X, Y = np.meshgrid(x,y)
z = (X**2+Y**2)/(X**2 + X*Y + Y**2)
n = 10
title = "Surface around limit 2"
zlabel = "z"
contlabel = "Contour around limit 2"
plot_task1("limit2")

# Limit 3, the limit is around -0.168
x = np.linspace(-1,1,1000)
y = np.linspace(-2,0,1000)

# x = np.linspace(-0.5,0.5,1000)
# y = np.linspace(-1.5,-0.5,1000)

# x = np.linspace (-0.1,0.1,1000)
# y = np.linspace(-1.1,-0.9,1000)

# Several linspaces are used to investigate an anomaly at the limit
# caused by floating point errors

X, Y = np.meshgrid(x,y)
z = (np.sin(X + X*Y) - X - X*Y)/((X*(Y+1))**3)
n = 10
title = "Surface around limit 3"
zlabel = "z"
contlabel = "Contour around limit 3"
plot_task1("limit3")

# The functions in (2)

#f(x,y) has local maxima of 4 at (1,2)
x = np.linspace(0, 2, 1000)
y = np.linspace(0, 4, 1000)
X, Y = np.meshgrid(x, y)
z = 8*X*Y - 4*Y*X**2 - 2*X*Y**2 + (X*Y)**2
n=10
title = "f(x,y) as a surface"
zlabel = "f(x,y)"
contlabel = "Contour of f(x,y) around local maxima"
plot_task1("f")

# g(x,y) has the local maxima of 1.2 at both (0,1) and (0,-1)
x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
X, Y = np.meshgrid(x, y)
z = (X**2 + 3*Y**2)*np.exp(-X**2-Y**2)
n=10
title = "g(x,y) as a surface"
zlabel = "g(x,y)"
contlabel = "Contour of g(x,y) around local maxima"
plot_task1("g")


plt.show()

# =============================================================================
# Task 2
# =============================================================================

print_header("Task 2: Gradients and Hessians")

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


# =============================================================================
# Task 3
# =============================================================================

print_header("Task 3: Implicit functions")

#gradient() and Hessian() from Task 2 use the point (a,b), so to reuse them
#at another point we first move (a,b) to that point
def gradient_at(f,x,y):
    global a, b
    a, b = x, y
    return gradient(f)

def Hessian_at(f,x,y):
    global a, b
    a, b = x, y
    return Hessian(f)


def F(x,y,z):
    return x+2*y+z+np.exp(2*z)-1

#Z(x,y) is the z that solves F(x,y,z) = 0, found with fsolve from z0 = 0
def Z(x,y):
    return fsolve(lambda z: F(x,y,z), 0)[0]

x = np.linspace(-0.5, 0.5, 41)
y = np.linspace(-0.5, 0.5, 41)
X, Y = np.meshgrid(x,y)
#fsolve solves for one point at a time, so we loop over the grid
z = np.zeros(X.shape)
for i in range(len(y)):
    for j in range(len(x)):
        z[i,j] = Z(X[i,j], Y[i,j])

fig3 = plt.figure()
ax3 = fig3.add_subplot(projection='3d')
ax3.plot_surface(X,Y,z,alpha=0.5)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Z(x,y)')
ax3.set_title('Z(x,y) defined by F(x,y,Z(x,y)) = 0')
save_figure(fig3, "task3_Z")

#Coefficients of the Taylor polynomial of Z at (0,0), numerical differentiation from Task 2
Z0 = Z(0,0)
dZ = gradient_at(Z,0,0)
hZ = Hessian_at(Z,0,0)
print("Taylor coefficients of Z at (0,0):")
print("Z(0,0)", Z0)
print("gradient of Z at (0,0)", dZ)
print("hessian of Z at (0,0)", hZ)

def P2(x,y):
    return Z0 + dZ[0]*x + dZ[1]*y + (hZ[0,0]*x**2 + 2*hZ[0,1]*x*y + hZ[1,1]*y**2)/2

p2 = P2(X,Y)

fig4 = plt.figure()
ax4 = fig4.add_subplot(projection='3d')
ax4.plot_surface(X,Y,p2,alpha=0.5)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('P2(x,y)')
ax4.set_title('Taylor polynomial P2(x,y) of Z at (0,0)')
save_figure(fig4, "task3_P2")

# The plots of Z and P2 look almost the same

e = np.abs(z-p2)

fig5 = plt.figure()
ax5 = fig5.add_subplot(projection='3d')
ax5.plot_surface(X,Y,e,alpha=0.5)
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_zlabel('e(x,y)')
ax5.set_title('Absolute error e(x,y) = |Z(x,y) - P2(x,y)|')
save_figure(fig5, "task3_error")

# The error is reasonable. It is 0 at (0,0) and grows when we go away from
# (0,0), like the distance cubed, since the rest term of a Taylor polynomial
# of degree 2 is of order |(x,y)|^3. It is largest in the corners (0.5,0.5)
# and (-0.5,-0.5), about 0.065.
# Sources of error:
# The Taylor polynomial stops at degree 2 (the rest term), this is the biggest part.
# The coefficients come from numerical differentiation with h = 1e-5, so they are not exact.
# fsolve only solves F = 0 up to a small tolerance.


# =============================================================================
# Task 4
# =============================================================================

print_header("Task 4: Himmelblau's function and gradient descent")

def himmelblau(x,y):
    return (x**2+y-11)**2 + (x+y**2-7)**2

x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x,y)
z = himmelblau(X,Y)
#levels only up to 250 so that the minima and the maximum (about 181) can be seen
n = np.linspace(0, 250, 26)

fig6 = plt.figure()
ax6 = fig6.add_subplot(projection='3d')
ax6.plot_surface(X,Y,z,alpha=0.5)
ax6.set_xlabel('x')
ax6.set_ylabel('y')
ax6.set_zlabel('f(x,y)')
ax6.set_title("Himmelblau's function")
save_figure(fig6, "task4_surface")

fig7, ax7 = plt.subplots()
cs = ax7.contourf(X,Y,z,n,extend='max')
cbar = fig7.colorbar(cs)
cbar.set_label('f(x,y)')
ax7.set_xlabel('x')
ax7.set_ylabel('y')
ax7.set_title("Himmelblau's function with the gradient descent iterates")

# f(x,y) has 4 local minima (all with f = 0) at about (3,2), (-2.8,3.1),
# (-3.8,-3.3) and (3.6,-1.8), and 1 local maximum of about 181.6 at (-0.27,-0.92)

#Implementation of gradient descent, equation (6), with the gradient from Task 2
def gradient_descent(f,x0,y0,alpha=0.01,steps=20):
    points = [np.array([x0,y0])]
    for k in range(steps):
        p = points[-1]
        points.append(p - alpha*gradient_at(f,p[0],p[1]))
    return np.array(points)

for start in [(0,0), (1/5,-4)]:
    points = gradient_descent(himmelblau, start[0], start[1])
    ax7.plot(points[:,0], points[:,1], 'o-', ms=3, label='start ' + str(start))
    x20 = points[-1]
    print("\nstart", start, "x20", x20)
    print("gradient at x20", gradient_at(himmelblau, x20[0], x20[1]))
    print("hessian at x20", Hessian_at(himmelblau, x20[0], x20[1]))

ax7.legend()
save_figure(fig7, "task4_contour")

# Yes, from (0,0) it goes to the minimum at (3,2) and from (1/5,-4) to the
# minimum at about (3.58,-1.85). The gradient at x20 is small but not exactly 0,
# since 20 steps is not enough to get all the way. The determinant of the
# hessian is positive and f_xx is positive, which says that it is a minimum.

# To find local maxima we take the step in the direction of the gradient instead,
# x_k+1 = x_k + alpha*grad f(x_k)

plt.show(block=True)
