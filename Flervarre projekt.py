import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
fig2, ax2 = plt.subplots()

# Task 1

# To view a limit just uncomment it and then run the code
# This was done to make it easier to read the result from the question asked

# All of the limits in (1)

# Limit 1, the limit is 1/2
# x = y = np.linspace(0, 2, 1000)
# X, Y = np.meshgrid(x,y)
# z = (X**2-X*Y)/(X**2-Y**2)
# n = 10
# title = "Surface around the limit"
# zlabel = "z"
# contlabel = "Contour around the limit"

# Limit 2, the limit does not exist
# x = y = np.linspace(-1, 1, 1000)
# X, Y = np.meshgrid(x,y)
# z = (X**2+Y**2)/(X**2 + X*Y + Y**2)
# n = 10
# title = "Surface around the limit"
# zlabel = "z"
# contlabel = "Contour around the limit"

# Limit 3, the limit is around -0.168
# x = np.linspace(-1,1,1000)
# y = np.linspace(-2,0,1000)

# x = np.linspace(-0.5,0.5,1000)
# y = np.linspace(-1.5,-0.5,1000)

# x = np.linspace (-0.1,0.1,1000)
# y = np.linspace(-1.1,-0.9,1000)

# Several linspaces are used to investigate an anomaly at the limit
# caused by floating point errors

# X, Y = np.meshgrid(x,y)
# z = (np.sin(X + X*Y) - X - X*Y)/((X*(Y+1))**3)
# n = 10
# title = "Surface around the limit"
# zlabel = "z"
# contlabel = "Contour around the limit"

# The functions in (2)

#f(x,y) has local maxima of 4 at (1,2)
# x = np.linspace(0, 2, 1000)
# y = np.linspace(0, 4, 1000)
# X, Y = np.meshgrid(x, y)
# z = 8*X*Y - 4*Y*X**2 - 2*X*Y**2 + (X*Y)**2
# n=10
# title = "f(x,y) as a surface"
# zlabel = "f(x,y)"
# contlabel = "Contour of f(x,y) around local maxima"

# g(x,y) has the local maxima of 1.2 at both (0,1) and (0,-1)
# x = np.linspace(-2, 2, 1000)
# y = np.linspace(-2, 2, 1000)
# X, Y = np.meshgrid(x, y)
# z = (X**2 + 3*Y**2)*np.exp(-X**2-Y**2)
# n=10
# title = "g(x,y) as a surface"
# zlabel = "g(x,y)"
# contlabel = "Contour of g(x,y) around local maxima"

# This plotting is universal for all limits and functions in task 1
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


plt.show()
