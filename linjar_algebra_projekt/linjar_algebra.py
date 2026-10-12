import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

# the data files are read from the same folder as this script
script_folder = os.path.dirname(os.path.abspath(__file__))


# Task 1

identity = np.identity(20, dtype = float)
identity2 = np.identity(11, dtype = float)
folder_path = os.path.join(script_folder, "regression_1.dat")

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







folder_path = os.path.join(script_folder, "regression_2.dat")

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


# Task 2

A = np.array([[1, 3, 2],
              [-3, 4, 3],
              [2, 3, 1]], dtype = float)
N = 400


# Task 2.1

eigenvalues, eigenvectors = scipy.linalg.eig(A)
print(eigenvalues)
print(eigenvectors)

# scipy returns complex numbers, but the imaginary parts are all 0
eigenvalues = eigenvalues.real

# check that A e = lambda e for every column e
i = 0
while i<3:
    print(eigenvalues[i], np.allclose(np.dot(A, eigenvectors[:, i]), eigenvalues[i]*eigenvectors[:, i]))
    i+=1

# the same eigenvectors, scaled by hand so that all entries are integers
e1 = np.array([3, 1, 3], dtype = float)
e2 = np.array([1, 0, 1], dtype = float)
e3 = np.array([1, 12, -19], dtype = float)
lambda1 = 4.0
lambda2 = 3.0
lambda3 = -1.0

print(np.dot(A, e1) == lambda1*e1, np.dot(A, e2) == lambda2*e2, np.dot(A, e3) == lambda3*e3)

# A = E D E^-1 with the eigenvectors as columns in E
E = np.array([e1, e2, e3]).T
D = np.diag([lambda1, lambda2, lambda3])
E_inverse = np.linalg.inv(E)
print(np.allclose(A, np.dot(E, np.dot(D, E_inverse))))

# the initial values, stored in a list so that we can loop over them
alphas = [1, 1/19]
alpha_names = ["1", "1/19"]
z0_list = [np.array([9, 1, 9], dtype = float)]
z0_names = ["[9, 1, 9]"]
i = 0
while i<2:
    z0_list.append(alphas[i]*np.array([1, 0, 1], dtype = float))
    z0_names.append(alpha_names[i] + "*[1, 0, 1]")
    z0_list.append(alphas[i]*np.array([1, 12, -19], dtype = float))
    z0_names.append(alpha_names[i] + "*[1, 12, -19]")
    i+=1

# z0 = c1*e1 + c2*e2 + c3*e3, so c = E^-1 z0
i = 0
while i<len(z0_list):
    c = np.dot(E_inverse, z0_list[i])
    print("z0 =", z0_names[i], " c =", c)
    i+=1

# z_n = A^n z0 = E D^n E^-1 z0 = c1*4^n*e1 + c2*3^n*e2 + c3*(-1)^n*e3
def closed_form(z0, n):
    c = np.dot(E_inverse, z0)
    return c[0]*lambda1**n*e1 + c[1]*lambda2**n*e2 + c[2]*lambda3**n*e3


# Task 2.2

def iterate(z0, N, A):
    z_list = [z0]
    n = 0
    while n<N:
        z_list.append(np.dot(A, z_list[n]))
        n+=1
    return z_list

# z/||z||. We divide by the largest entry first, since ||z||^2 is too large
# for a float when z grows like 4^n (n > 255)
def direction(z):
    z = z/np.max(np.abs(z))
    return z/np.linalg.norm(z)

i = 0
while i<len(z0_list):
    z_list = iterate(z0_list[i], N, A)
    print("z0 =", z0_names[i])
    print("z_400 direction:", direction(z_list[N]))
    print("z_399 direction:", direction(z_list[N-1]))
    print("closed form, z_400 direction:", direction(closed_form(z0_list[i], N)))
    print("closed form, z_399 direction:", direction(closed_form(z0_list[i], N-1)))
    i+=1


# Task 2.3

# The term with the largest |lambda| among those with c != 0 takes over,
# so v_n -> sign(c)*e/||e|| and q_n -> lambda for that eigenvalue.
# If lambda = -1 the sign switches every step, then v is the limit for even n.
def theoretical_limit(z0):
    c = np.dot(E_inverse, z0)
    if abs(c[0])>1e-10:
        return np.sign(c[0])*e1/np.linalg.norm(e1), lambda1
    elif abs(c[1])>1e-10:
        return np.sign(c[1])*e2/np.linalg.norm(e2), lambda2
    else:
        return np.sign(c[2])*e3/np.linalg.norm(e3), lambda3


# Task 2.4 and 2.5

# v_0 = z0/||z0||, v_n = A v_(n-1)/||A v_(n-1)|| and q_n = v_n^T A v_n
def normalized_iterate(z0, N, A):
    v_list = [z0/np.linalg.norm(z0)]
    q_list = [np.dot(v_list[0], np.dot(A, v_list[0]))]
    n = 0
    while n<N:
        w = np.dot(A, v_list[n])
        v_list.append(w/np.linalg.norm(w))
        q_list.append(np.dot(v_list[n+1], np.dot(A, v_list[n+1])))
        n+=1
    return v_list, q_list

all_v = []
all_q = []
all_v_theory = []
all_q_theory = []
i = 0
while i<len(z0_list):
    v_list, q_list = normalized_iterate(z0_list[i], N, A)
    v_theory, q_theory = theoretical_limit(z0_list[i])
    all_v.append(v_list)
    all_q.append(q_list)
    all_v_theory.append(v_theory)
    all_q_theory.append(q_theory)
    print("z0 =", z0_names[i])
    print("v_400:", v_list[N])
    print("v_399:", v_list[N-1])
    print("v theory:", v_theory)
    print("q_400:", q_list[N], " q_399:", q_list[N-1], " q theory:", q_theory)
    i+=1


# Task 2.6

# x_n = 1 + 1/2 + ... + 1/n, the steps get small but x_n grows like ln(n)
x = 0
n = 1
while n<=10**6:
    x += 1/n
    if n==10 or n==1000 or n==10**6:
        print("n =", n, " x_n =", x, " ln(n) =", np.log(n), " x_n - x_(n-1) =", 1/n, " x_(n+1) - x_(n-1) =", 1/n + 1/(n+1))
    n+=1


# Task 2.7

def error_list(v_list, v_star):
    errors = []
    n = 0
    while n<len(v_list):
        errors.append(np.linalg.norm(v_list[n] - v_star))
        n+=1
    return errors

# smallest n such that errors[m] < eps for all m >= n (up to N),
# None if the error is not below eps at the end
def iterations_needed(errors, eps):
    n = len(errors) - 1
    if errors[n]>=eps:
        return None
    while n>0 and errors[n-1]<eps:
        n -= 1
    return n

print("z0, v* = v, v* = v_400, q* = q, q* = q_400")
i = 0
while i<len(z0_list):
    print(z0_names[i],
          iterations_needed(error_list(all_v[i], all_v_theory[i]), 1e-8),
          iterations_needed(error_list(all_v[i], all_v[i][N]), 1e-8),
          iterations_needed(error_list(all_q[i], all_q_theory[i]), 1e-8),
          iterations_needed(error_list(all_q[i], all_q[i][N]), 1e-8))
    i+=1


# Task 2.8

eps_list = np.logspace(-1, -14, 14)

def plot_iterations(errors, name):
    n_plot = []
    eps_plot = []
    for eps in eps_list:
        n = iterations_needed(errors, eps)
        if n != None:
            n_plot.append(n)
            eps_plot.append(eps)
    plt.semilogy(n_plot, eps_plot, "o-", label=name)

i = 0
while i<len(z0_list):
    plot_iterations(error_list(all_v[i], all_v_theory[i]), "v* = v")
    plot_iterations(error_list(all_v[i], all_v[i][N]), "v* = v_400")
    plot_iterations(error_list(all_q[i], all_q_theory[i]), "q* = q")
    if all_q[i][N] != all_q_theory[i]:
        plot_iterations(error_list(all_q[i], all_q[i][N]), "q* = q_400")
    plt.title("z0 = " + z0_names[i])
    plt.xlabel("n")
    plt.ylabel("epsilon")
    plt.legend()
    plt.show()
    i+=1
