# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:12:05 2026
@author: Folke Adolfsson
"""
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
fig2, ax2 = plt.subplots()

# Task 1

# All of the limits in (1)

# Limit 1, the limit is 1/2
# x = y = np.linspace(0, 2, 1000)
# X, Y = np.meshgrid(x,y)
# z = (X**2-X*Y)/(X**2-Y**2)
# n = 10

# Limit 2, the limit does not exist
# x = y = np.linspace(-1, 1, 1000)
# X, Y = np.meshgrid(x,y)
# z = (X**2+Y**2)/(X**2 + X*Y + Y**2)
# n = 10

# Limit 3, 
# x = np.linspace(-1,1,1000)
# y = np.linspace(-2,0,1000)
# X, Y = np.meshgrid(x,y)
# z = (np.sin(X + X*Y) - X - X*Y)/((X*(Y+1))**3)
# n = 20

# The functions in (2)

# f(x,y) has local maxima of 4 at (1,2)
# x = np.linspace(0, 2, 1000)
# y = np.linspace(0, 4, 1000)
# X, Y = np.meshgrid(x, y)
# z = 8*X*Y - 4*Y*X**2 - 2*X*Y**2 + (X*Y)**2
# n=10

# g(x,y) has the local maxima of 1.2 at both (0,1) and (0,-1)
x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
X, Y = np.meshgrid(x, y)
z = (X**2 + 3*Y**2)*np.exp(-X**2-Y**2)
n=10

ax.plot_surface(X,Y,z,alpha=0.5)
cs = ax2.contourf(X,Y,z,n)
cbar = fig2.colorbar(cs)

plt.show()