import matplotlib.pyplot as plt
import numpy as np

identity = np.identity(20, dtype = float)
identity2 = np.identity(11, dtype = float)
folder_path = r"C:\Users\Bruno\Downloads\regression_1-1.dat"

with open(folder_path,"r") as file:
    lines_list = file.readlines()
    X_1 = []
    xcoord = []
    ycoord = []
    i = 0
    while i<20:
        X_1.append(lines_list[i+1].split(","))
        xcoord.append(float(X_1[i][0]))
        ycoord.append(float(X_1[i][1]))
        X_1[i][1] = 1
        X_1[i][0] = float(X_1[i][0])
        i+=1
    file.close()
    

A = np.array(X_1)
Y = np.array(ycoord)
A_transpose = A.T
AtA = np.dot(A_transpose,A)
AtA_inverse = (1/((AtA[0][0])*(AtA[1][1])-(AtA[0][1])*(AtA[1][0])))*(np.array([[AtA[1][1],-AtA[0][1]],[-AtA[1][0],AtA[0][0]]]))
solution = np.dot(AtA_inverse, np.dot(A_transpose, Y))
print(solution)

P = np.dot(A,np.dot(AtA_inverse,A_transpose))
R = 2*P-identity

print(np.allclose(np.dot(R,R),identity))

PY = np.dot(P,Y.reshape(-1,1))
RY = np.dot(R,Y.reshape(-1,1))


X = np.arange(-6,6,0.05)
plt.plot(xcoord,ycoord, "o")
plt.plot(X,X*solution[0]+solution[1])
plt.show()

plt.plot(xcoord, PY, "o", label="Projection")
plt.plot(xcoord, RY, "o", label="Reflection")
plt.plot(xcoord, ycoord, "o", label="Original data")
plt.plot(X, X*solution[0] + solution[1], label="Regression line")
plt.legend()
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







folder_path2 = folder_path = r"C:\Users\Bruno\Downloads\regression_2-1.dat"

with open(folder_path,"r") as file:
    lines_list = file.readlines()
    X2_1 = []
    x2coord = []
    y2_1coord = []
    y2_2coord = []
    y2_3coord = []
    i = 0
    while i<11:
        X2_1.append(lines_list[i+1].split(","))
        x2coord.append(float(X2_1[i][0]))
        y2_1coord.append(float(X2_1[i][1]))
        y2_2coord.append(float(X2_1[i][2]))
        y2_3coord.append(float(X2_1[i][3]))
        del X2_1[i][3]
        del X2_1[i][2]
        X2_1[i][1] = 1
        X2_1[i][0] = float(X2_1[i][0])
        i+=1
    file.close()
    
A2 = np.array(X2_1)
Y2_1 = np.array(y2_1coord)
A_transpose = A2.T
AtA = np.dot(A_transpose,A2)
AtA_inverse = (1/((AtA[0][0])*(AtA[1][1])-(AtA[0][1])*(AtA[1][0])))*(np.array([[AtA[1][1],-AtA[0][1]],[-AtA[1][0],AtA[0][0]]]))
solution2_1 = np.dot(AtA_inverse, np.dot(A_transpose, Y2_1))
print(solution2_1)

X = np.arange(2.5,15,0.05)
plt.plot(x2coord,y2_1coord, "o")
plt.plot(X,X*solution2_1[0]+solution2_1[1])
plt.show()

Y2_2 = np.array(y2_2coord)
solution2_2 = np.dot(AtA_inverse, np.dot(A_transpose, Y2_2))
print(solution2_2)

X = np.arange(2.5,15,0.05)
plt.plot(x2coord,y2_2coord, "o")
plt.plot(X,X*solution2_2[0]+solution2_2[1])
plt.show()

Y2_3 = np.array(y2_3coord)
solution2_3 = np.dot(AtA_inverse, np.dot(A_transpose, Y2_3))
print(solution2_3)

X = np.arange(2.5,15,0.05)
plt.plot(x2coord,y2_3coord, "o")
plt.plot(X,X*solution2_3[0]+solution2_3[1])
plt.show()

P2 = np.dot(A2,np.dot(AtA_inverse,A_transpose))
R2 = 2*P2-identity2

def check_projection(P):
    return np.allclose(P, np.dot(P,P)), np.allclose(P, P.T)

print(check_projection(P2))


# verifying that y_hat is the result of an orthogonal projection
y_hat = xcoord*solution[0]+solution[1]
error = Y-y_hat

print(np.dot(y_hat, error))






