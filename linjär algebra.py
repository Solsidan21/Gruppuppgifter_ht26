import matplotlib.pyplot as plt
import numpy as np

folder_path = r"C:\Users\Bruno\Downloads\regression_1-1.dat"

with open(folder_path,"r") as file:
    lines_list = file.readlines()
    q = []
    xcoord = []
    ycoord = []
    i = 0
    while i<20:
        q.append(lines_list[i+1].split(","))
        xcoord.append(float(q[i][0]))
        ycoord.append(float(q[i][1]))
        q[i][1] = 1
        q[i][0] = float(q[i][0])
        i+=1
    file.close()
    

A = np.array(q)
Y = np.array(ycoord)
A_transpose = A.T
AtA = np.dot(A_transpose,A)
AtA_inverse = (1/((AtA[0][0])*(AtA[1][1])-(AtA[0][1])*(AtA[1][0])))*(np.array([[AtA[1][1],-AtA[0][1]],[-AtA[1][0],AtA[0][0]]]))
solution = np.dot(AtA_inverse, np.dot(A_transpose, Y))
print(solution)


X = np.arange(-6,6,0.05)
plt.plot(xcoord,ycoord, "o")
plt.plot(X,X*solution[0]+solution[1])
plt.show()



folder_path2 = folder_path = r"C:\Users\Bruno\Downloads\regression_2-1.dat"

with open(folder_path,"r") as file:
    lines_list = file.readlines()
    q = []
    x1coord = []
    y1coord = []
    y2coord = []
    y3coord = []
    i = 0
    while i<11:
        q.append(lines_list[i+1].split(","))
        x1coord.append(float(q[i][0]))
        y1coord.append(float(q[i][1]))
        y2coord.append(float(q[i][2]))
        y3coord.append(float(q[i][3]))
        del q[i][3]
        del q[i][2]
        q[i][1] = 1
        q[i][0] = float(q[i][0])
        i+=1
    file.close()
    
A2 = np.array(q)
Y2 = np.array(y3coord)
A_transpose = A2.T
AtA = np.dot(A_transpose,A2)
AtA_inverse = (1/((AtA[0][0])*(AtA[1][1])-(AtA[0][1])*(AtA[1][0])))*(np.array([[AtA[1][1],-AtA[0][1]],[-AtA[1][0],AtA[0][0]]]))
solution2 = np.dot(AtA_inverse, np.dot(A_transpose, Y2))
print(solution2)

X = np.arange(2.5,15,0.05)
plt.plot(x1coord,y3coord, "o")
plt.plot(X,X*solution2[0]+solution2[1])
plt.show()

a = np.linspace(0, 2, 200)
b = np.linspace(0, 2, 200)
a, b = np.meshgrid(a, b)

xcoord = np.array(xcoord)
ycoord = np.array(ycoord)

# Shape: (20, 200, 200), help from chatgpt to understand format for plot_surface
errors = xcoord[:, None, None] * a + b - ycoord[:, None, None]

# Sum over the 20 data points
z = np.sqrt(np.sum(errors**2, axis=0))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(a, b, z, alpha=0.5)

ax.plot(solution[0], solution[1],np.sqrt(sum((solution[0]*xcoord+solution[1]-ycoord)**2)) , marker='o')
ax.view_init(elev=0, azim=90)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('z')
plt.show()


x1coord = np.array(x1coord)
x1coord = x1coord.reshape(-1, 1)
xt = x1coord.T
inv = 1/(np.dot(xt,x1coord))
P1 = inv*np.dot(x1coord,xt)
Check = np.allclose(P1,np.dot(P1,P1))
print(Check)

xcoord = np.array(xcoord)
xcoord = xcoord.reshape(-1, 1)
xt = xcoord.T
inv = 1/(np.dot(xt,xcoord))
P2 = inv*np.dot(xcoord,xt)
Check = np.allclose(P2,np.dot(P2,P2))
print(Check)

print(np.allclose(P1,P1.T))
print(np.allclose(P2,P2.T))
identity = np.identity(20, dtype = float)
PY = np.dot(P2,Y.reshape(-1,1))

plt.plot(xcoord*solution[0]+solution[1],PY, "o")
plt.plot()
plt.show()



# show that y and ^y are orthogonal and a direct sum to an inner product space 
Y_hatt_rowvector = (xcoord*solution[0]+solution[1]).T
Yreshape = Y.reshape(-1,1)
degree = np.dot(Y_hatt_rowvector,Yreshape-(xcoord*solution[0]+solution[1]))

print(degree)

rank = np.linalg.matrix_rank(np.row_stack((Y, Y_hatt_rowvector)))

print(rank)






