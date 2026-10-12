import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.optimize

# the data files are read from the same folder as this script,
# and all figures are saved as pdf in the subfolder figures
script_folder = os.path.dirname(os.path.abspath(__file__))
figure_folder = os.path.join(script_folder, "figures")
os.makedirs(figure_folder, exist_ok=True)

def save_figure(name):
    plt.savefig(os.path.join(figure_folder, name + ".pdf"), bbox_inches="tight")


# Task 1

data1 = np.loadtxt(os.path.join(script_folder, "regression_1.dat"), delimiter=",", skiprows=1)
data2 = np.loadtxt(os.path.join(script_folder, "regression_2.dat"), delimiter=",", skiprows=1)

# the four datasets as [name, x, y]
datasets = [["regression_1", data1[:, 0], data1[:, 1]],
            ["regression_2_y1", data2[:, 0], data2[:, 1]],
            ["regression_2_y2", data2[:, 0], data2[:, 2]],
            ["regression_2_y3", data2[:, 0], data2[:, 3]]]

# ||ax + b - y||^2, the function that fmin minimizes
def squared_error(ab, x, y):
    return np.sum((ab[0]*x + ab[1] - y)**2)

def check_projection(P):
    return np.allclose(P, P @ P), np.allclose(P, P.T)

def check_reflection(R):
    identity = np.eye(len(R))
    return np.allclose(R @ R, identity), np.allclose(R, R.T)

for k in range(len(datasets)):
    name, x, y = datasets[k]
    print(name)

    # Task 1.1
    A = np.column_stack((x, np.ones(len(x))))
    solution = np.linalg.solve(A.T @ A, A.T @ y)
    solution_fmin = scipy.optimize.fmin(squared_error, [0, 0], args=(x, y), xtol=1e-10, ftol=1e-10, maxiter=10000, maxfun=10000, disp=False)
    print("normal equations: a, b =", solution)
    print("fmin:             a, b =", solution_fmin)

    # Task 1.2
    X = np.linspace(min(x) - 1, max(x) + 1, 200)
    plt.plot(x, y, "o", label="Original data")
    plt.plot(X, X*solution[0] + solution[1], label="Regression line")
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    save_figure("task1_fit_" + name)
    plt.show()

    # Task 1.5, 1.6, 1.8 and 1.9
    P = A @ np.linalg.solve(A.T @ A, A.T)
    R = 2*P - np.eye(len(y))
    y_hat = P @ y
    Ry = R @ y
    print("Py = A*solution:", np.allclose(y_hat, A @ solution))
    print("P projection, P orthogonal:", check_projection(P))
    # y_hat is an orthogonal projection of y if y_hat is orthogonal to y - y_hat
    print("<y_hat, y - y_hat> =", y_hat @ (y - y_hat))
    print("R reflection, R orthogonal:", check_reflection(R))

    # Task 1.7 and 1.9
    plt.plot(x, Ry, "o", label="Reflection Ry")
    plt.plot(x, y_hat, "o", label="Projection Py")
    plt.plot(x, y, "o", label="Original data")
    plt.plot(X, X*solution[0] + solution[1], label="Regression line")
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    save_figure("task1_projection_" + name)
    plt.show()

    # Task 1.3
    a = np.linspace(solution[0] - 1, solution[0] + 1, 200)
    b = np.linspace(solution[1] - 3, solution[1] + 3, 200)
    a, b = np.meshgrid(a, b)
    # Shape: (n, 200, 200), help from chatgpt to understand format for plot_surface
    errors = x[:, None, None]*a + b - y[:, None, None]
    # Sum over the n data points
    z = np.sqrt(np.sum(errors**2, axis=0))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(a, b, z, alpha=0.5)
    ax.plot(solution[0], solution[1], np.linalg.norm(A @ solution - y), "o", color="red")
    ax.set_title(name)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("error")
    save_figure("task1_error_surface_" + name)
    plt.show()


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
for i in range(3):
    print(eigenvalues[i], np.allclose(A @ eigenvectors[:, i], eigenvalues[i]*eigenvectors[:, i]))

# the same eigenvectors, scaled by hand so that all entries are integers
e1 = np.array([3, 1, 3], dtype = float)
e2 = np.array([1, 0, 1], dtype = float)
e3 = np.array([1, 12, -19], dtype = float)
lambda1 = 4.0
lambda2 = 3.0
lambda3 = -1.0

print(A @ e1 == lambda1*e1, A @ e2 == lambda2*e2, A @ e3 == lambda3*e3)

# A = E D E^-1 with the eigenvectors as columns in E
E = np.array([e1, e2, e3]).T
D = np.diag([lambda1, lambda2, lambda3])
print(np.allclose(A @ E, E @ D))

# the initial values, stored in a list so that we can loop over them
z0_list = [np.array([9, 1, 9], dtype = float)]
z0_names = ["[9, 1, 9]"]
for alpha, alpha_name in [[1, "1"], [1/19, "1/19"]]:
    z0_list.append(alpha*np.array([1, 0, 1], dtype = float))
    z0_names.append(alpha_name + "*[1, 0, 1]")
    z0_list.append(alpha*np.array([1, 12, -19], dtype = float))
    z0_names.append(alpha_name + "*[1, 12, -19]")

# z0 = c1*e1 + c2*e2 + c3*e3, so c solves E c = z0
for i in range(len(z0_list)):
    c = np.linalg.solve(E, z0_list[i])
    print("z0 =", z0_names[i], " c =", c)

# z_n = A^n z0 = E D^n E^-1 z0 = c1*4^n*e1 + c2*3^n*e2 + c3*(-1)^n*e3
def closed_form(z0, n):
    c = np.linalg.solve(E, z0)
    return c[0]*lambda1**n*e1 + c[1]*lambda2**n*e2 + c[2]*lambda3**n*e3


# Task 2.2

# z_0, z_1, ..., z_N as the rows of Z
def iterate(z0, N, A):
    Z = np.zeros((N + 1, len(z0)))
    Z[0] = z0
    for n in range(N):
        Z[n + 1] = A @ Z[n]
    return Z

# z/||z||. We divide by the largest entry first, since ||z||^2 is too large
# for a float when z grows like 4^n (n > 255)
def direction(z):
    z = z/np.max(np.abs(z))
    return z/np.linalg.norm(z)

for i in range(len(z0_list)):
    Z = iterate(z0_list[i], N, A)
    print("z0 =", z0_names[i])
    print("z_400 direction:", direction(Z[N]))
    print("z_399 direction:", direction(Z[N - 1]))
    print("closed form, z_400 direction:", direction(closed_form(z0_list[i], N)))
    print("closed form, z_399 direction:", direction(closed_form(z0_list[i], N - 1)))


# Task 2.3

# The term with the largest |lambda| among those with c != 0 takes over,
# so v_n -> sign(c)*e/||e|| and q_n -> lambda for that eigenvalue.
# If lambda = -1 the sign switches every step, then v is the limit for even n.
def theoretical_limit(z0):
    c = np.linalg.solve(E, z0)
    if abs(c[0]) > 1e-10:
        return np.sign(c[0])*e1/np.linalg.norm(e1), lambda1
    elif abs(c[1]) > 1e-10:
        return np.sign(c[1])*e2/np.linalg.norm(e2), lambda2
    else:
        return np.sign(c[2])*e3/np.linalg.norm(e3), lambda3


# Task 2.4 and 2.5

# v_0 = z0/||z0||, v_n = A v_(n-1)/||A v_(n-1)|| and q_n = v_n^T A v_n
def normalized_iterate(z0, N, A):
    V = np.zeros((N + 1, len(z0)))
    q = np.zeros(N + 1)
    V[0] = z0/np.linalg.norm(z0)
    q[0] = V[0] @ A @ V[0]
    for n in range(N):
        w = A @ V[n]
        V[n + 1] = w/np.linalg.norm(w)
        q[n + 1] = V[n + 1] @ A @ V[n + 1]
    return V, q

all_V = []
all_q = []
all_v_theory = []
all_q_theory = []
for i in range(len(z0_list)):
    V, q = normalized_iterate(z0_list[i], N, A)
    v_theory, q_theory = theoretical_limit(z0_list[i])
    all_V.append(V)
    all_q.append(q)
    all_v_theory.append(v_theory)
    all_q_theory.append(q_theory)
    print("z0 =", z0_names[i])
    print("v_400:", V[N])
    print("v_399:", V[N - 1])
    print("v theory:", v_theory)
    print("q_400:", q[N], " q_399:", q[N - 1], " q theory:", q_theory)


# Task 2.6

# x_n = 1 + 1/2 + ... + 1/n, the steps get small but x_n grows like ln(n)
x = 0
for n in range(1, 10**6 + 1):
    x += 1/n
    if n == 10 or n == 1000 or n == 10**6:
        print("n =", n, " x_n =", x, " ln(n) =", np.log(n), " x_n - x_(n-1) =", 1/n, " x_(n+1) - x_(n-1) =", 1/n + 1/(n+1))


# Task 2.7

# smallest n such that errors[m] < eps for all m >= n (up to N),
# None if the error is not below eps at the end
def iterations_needed(errors, eps):
    n = len(errors) - 1
    if errors[n] >= eps:
        return None
    while n > 0 and errors[n - 1] < eps:
        n -= 1
    return n

print("z0, v* = v, v* = v_400, q* = q, q* = q_400")
for i in range(len(z0_list)):
    V = all_V[i]
    q = all_q[i]
    print(z0_names[i],
          iterations_needed(np.linalg.norm(V - all_v_theory[i], axis=1), 1e-8),
          iterations_needed(np.linalg.norm(V - V[N], axis=1), 1e-8),
          iterations_needed(np.abs(q - all_q_theory[i]), 1e-8),
          iterations_needed(np.abs(q - q[N]), 1e-8))


# Task 2.8

eps_list = np.logspace(-1, -14, 14)

def plot_iterations(errors, name):
    n_plot = []
    eps_plot = []
    for eps in eps_list:
        n = iterations_needed(errors, eps)
        if n is not None:
            n_plot.append(n)
            eps_plot.append(eps)
    plt.semilogy(n_plot, eps_plot, "o-", label=name)

for i in range(len(z0_list)):
    V = all_V[i]
    q = all_q[i]
    plot_iterations(np.linalg.norm(V - all_v_theory[i], axis=1), "v* = v")
    plot_iterations(np.linalg.norm(V - V[N], axis=1), "v* = v_400")
    plot_iterations(np.abs(q - all_q_theory[i]), "q* = q")
    if q[N] != all_q_theory[i]:
        plot_iterations(np.abs(q - q[N]), "q* = q_400")
    plt.title("z0 = " + z0_names[i])
    plt.xlabel("n")
    plt.ylabel("epsilon")
    plt.legend()
    save_figure("task2_iterations_" + str(i + 1))
    plt.show()
