import numpy as np
import matplotlib.pyplot as plt

# === Problem 2: Gradient Descent with Armijo Line Search ===

# Objective function
def f(xk):
    x1, x2 = xk[0], xk[1]
    return 0.26 * (x1**2 + x2**2) - 0.46 * x1 * x2

# Gradient of the function
def grad_f(xk):
    x1, x2 = xk[0], xk[1]
    return np.array([0.52 * x1 - 0.46 * x2,
                     -0.46 * x1 + 0.52 * x2])

# Armijo line search update
def armijo_update(xk, gamma0, alpha, beta):
    gamma = gamma0
    grad_J = grad_f(xk)
    zk = -grad_J
    while f(xk + gamma * zk) > f(xk) - alpha * gamma * grad_J @ zk:
        gamma *= beta
    return xk + gamma * zk

# Initialization
num_iter = 100
xk = np.array([-4.0, -2.0])
xk_list = [xk.copy()]
f_val_list = [f(xk)]

# Line search parameters
gamma0 = 1.0
alpha = 1e-4
beta = 0.5

# Run gradient descent with Armijo
for i in range(num_iter):
    xk = armijo_update(xk, gamma0, alpha, beta)
    xk_list.append(xk.copy())
    f_val_list.append(f(xk))

xk_list = np.array(xk_list)
xk_initial = xk_list[0]
xk_final = xk_list[-1]

# === Plot 1: Contour plot with descent path ===
x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = 0.26 * (X**2 + Y**2) - 0.46 * X * Y

plt.figure(figsize=(8, 6))
contours = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(xk_list[:, 0], xk_list[:, 1], 'ro-', markersize=3, label='Descent Path')
plt.plot(xk_initial[0], xk_initial[1], 'bo', label='Start Point')
plt.plot(xk_final[0], xk_final[1], 'go', label='End Point')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Problem 2: Gradient Descent with Armijo Line Search')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig("problem2_trajectory.png", dpi=300)
plt.show()

# === Plot 2: Objective function convergence ===
plt.figure(figsize=(6, 4))
plt.plot(f_val_list, 'm-', label='f(x)')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Convergence of Objective Function')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("problem2_convergence.png", dpi=300)
plt.show()
