"""
ME455 – HW5 – Problem 3 · Ergodic control for a differential‑drive robot
-----------------------------------------------------------------------
Target distribution & Fourier‑basis metric follow Problem 1/2.  The robot
state is [x, y, θ] with controls u = [v, ω] (linear & angular velocity).
Only the (x,y) coordinates are used inside the ergodic metric.

Output (saved as  *HW5_Problem3.png*):
  • left  : ergodic trajectory overlaid on Gaussian‑mixture density
  • centre: optimal controls v(t), ω(t)  (smoothed for readability)
  • right : objective value per iLQR iteration

The script is completely self‑contained; NumPy, SciPy, and Matplotlib are
the only requirements.  Runtime ≈ 1 – 2 minutes on a laptop CPU.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.stats import multivariate_normal

# ─────────────────────────── Target GMM ──────────────────────────────
w  = [0.5, 0.2, 0.3]
mu = [np.array([0.35, 0.38]),
      np.array([0.68, 0.25]),
      np.array([0.56, 0.64])]
S  = [np.array([[0.01 ,  0.004], [ 0.004, 0.01 ]]),
      np.array([[0.005, -0.003], [-0.003, 0.005]]),
      np.array([[0.008,  0.   ], [ 0.   , 0.004]])]

# Fourier basis – 25 × 25 = 625 coefficients
K       = 25
k_list  = np.array([[i, j] for i in range(K) for j in range(K)])  # (625,2)
phi     = np.array([
    sum(wi * np.exp(-0.5 * (np.pi*k) @ Si @ (np.pi*k)) * np.cos(np.pi*k @ mi)
        for wi, mi, Si in zip(w, mu, S))
    for k in k_list
])
Lambda  = (1 + np.linalg.norm(k_list, axis=1)**2)**(-1.5)          # weighting

# ─────────────────────────── Parameters ──────────────────────────────
dt, T   = 0.1, 10.0
N       = int(T / dt)

x0      = np.array([0.30, 0.30, np.pi/2])        # initial [x, y, theta]

n_x = 3              # state dimension
n_u = 2              # control dimension (v, ω)

Qz  = np.diag([1e-3, 1e-3, 5e-4])   # quadratic weight on z(t)
Ru  = np.diag([3e-3, 1e-3])          # control effort
Rv  = np.diag([1e-3, 5e-4])          # descent‑direction weight
BARR = 0.2                           # soft barrier strength on workspace

invRv = np.linalg.inv(Rv)

# ─────────────────────────── Dynamics helpers ───────────────────────

def step(x, u):
    """One Euler step of diff‑drive kinematics."""
    v, w = u
    xdot = np.array([
        v * np.cos(x[2]),
        v * np.sin(x[2]),
        w
    ])
    return x + dt * xdot


def simulate(x0, U):
    """Roll‑out full trajectory   X ∈ ℝ^{N×3}."""
    X = np.zeros((N, n_x))
    x = x0.copy()
    for t in range(N):
        X[t] = x
        x    = step(x, U[t])
    return X


# ─────────────────────── Ergodic‑metric building blocks ─────────────

def compute_a_k(X):
    """Fourier coefficients a_k of trajectory (position only)."""
    pos = X[:, :2]
    return np.cos(np.pi * pos @ k_list.T).sum(0) * dt / T


def grad_state(x, ck):
    """∂E/∂x  (gradient wrt full [x,y,θ]; θ‑component is zero)."""
    pos = x[:2]
    s   = -2 * Lambda * (ck - phi) * np.sin(np.pi * k_list @ pos)
    grad_pos = (np.pi * dt / T) * (s[:, None] * k_list).sum(0)

    # soft barrier on [0,1]² for x,y
    barr = np.where(pos < 0, 2*pos, 0.) + np.where(pos > 1, 2*(pos-1), 0.)
    grad_pos += BARR * barr

    return np.hstack([grad_pos, 0.0])


# ───────────────────────── One iLQR iteration ───────────────────────

def ilqr_iter(U):
    X  = simulate(x0, U)
    ck = compute_a_k(X)

    # Pre‑compute per‑step quantities
    a_list = np.zeros((N, n_x))           # ∂E/∂x
    b_list = np.zeros((N, n_u))           # ∂E/∂u
    A_list = np.zeros((N, n_x, n_x))
    B_list = np.zeros((N, n_x, n_u))

    for i in range(N):
        x, u = X[i], U[i]
        v, w = u
        th   = x[2]

        # dynamics linearisation f_x, f_u
        A = np.array([[0.0,       0.0, -v*np.sin(th)],
                       [0.0,       0.0,  v*np.cos(th)],
                       [0.0,       0.0,  0.0        ]])
        B = np.array([[np.cos(th), 0.0],
                      [np.sin(th), 0.0],
                      [0.0       , 1.0]])

        A_list[i] = A
        B_list[i] = B

        a_list[i] = grad_state(x, ck)
        b_list[i] = 2 * Ru @ u

    # ODE for [z; p] ∈ ℝ^{2·n_x}.  Piece‑wise constant coefficients.
    def zp_dyn(t_vec, zp_mat):
        out = np.empty_like(zp_mat)
        idx = np.minimum((t_vec / dt).astype(int), N-1)
        for col, j in enumerate(idx):
            A = A_list[j]
            B = B_list[j]
            M11 = A
            M22 = -A.T
            M12 = -B @ invRv @ B.T
            M21 = -Qz
            M   = np.block([[M11, M12],
                              [M21, M22]])
            v1  = -B @ invRv @ b_list[j]
            v2  = -a_list[j]
            v   = np.hstack([v1, v2])
            out[:, col] = M @ zp_mat[:, col] + v
        return out

    def bc(z0, zT):          # z(0) = 0 ,  p(T) = 0
        return np.hstack([z0[:n_x], zT[n_x:]])

    grid = np.linspace(0, (N-1)*dt, N)
    sol  = solve_bvp(zp_dyn, bc, grid,
                     np.zeros((2*n_x, N)), max_nodes=5*N)
    p_traj = sol.sol(grid)[n_x:].T        # (N × 3)

    # descent direction v(t)
    V = np.empty((N, n_u))
    for j in range(N):
        B = B_list[j]
        V[j] = -(invRv @ (B.T @ p_traj[j] + b_list[j]))
    return V, ck, X


# ───────────────────────── Optimisation loop ────────────────────────
rng   = np.random.default_rng(0)
U     = rng.normal(0, 0.4, size=(N, n_u))      # seed controls (v,ω)
loss  = []
alpha, beta = 1e-3, 0.8                        # Armijo parameters

for it in range(150):
    V, ck, X = ilqr_iter(U)
    J        = (Lambda * (ck - phi)**2).sum()
    loss.append(J)

    # ─ line search ─
    g = 1.0
    while True:
        U_new = U + g * V
        J_new = (Lambda * (compute_a_k(simulate(x0, U_new)) - phi)**2).sum()
        if J_new <= J - alpha*g*(V**2).sum():
            break
        g *= beta
        if g < 1e-6:
            break
    U = U_new

    # optional early stop
    if it > 5 and abs(loss[-1] - loss[-2]) < 5e-5:
        break

print(f"Finished after {len(loss)} iterations.  Final objective = {loss[-1]:.3e}")

# ─────────────────────────── Plot results ───────────────────────────
t = np.linspace(0, T, N)
X = simulate(x0, U)
pos = X[:, :2]

plt.figure(figsize=(15,4))

# (1) Trajectory & density
plt.subplot(1,3,1)
xx, yy = np.meshgrid(np.linspace(0,1,200), np.linspace(0,1,200))
Z = sum(wi*multivariate_normal(mi,Si).pdf(np.dstack((xx,yy)))
        for wi,mi,Si in zip(w,mu,S))
plt.contourf(xx, yy, Z, levels=20, cmap='Reds', alpha=0.7)
plt.plot(pos[:,0], pos[:,1], 'k-', lw=2)
plt.scatter(pos[:,0], pos[:,1], c='k', s=6, zorder=2)
plt.scatter(x0[0], x0[1], c='dodgerblue', s=80, zorder=3)
plt.title('Ergodic Trajectory'); plt.xlabel('X'); plt.ylabel('Y')
plt.axis('equal'); plt.xlim(0,1); plt.ylim(0,1)

# (2) Optimal controls (smoothed)

def smooth(v):
    return np.convolve(v, np.ones(3)/3, mode='same')

plt.subplot(1,3,2)
plt.plot(t, smooth(U[:,0]), label=r'$v(t)$')
plt.plot(t, smooth(U[:,1]), label=r'$\omega(t)$')
plt.title('Optimal Control'); plt.xlabel('Time'); plt.ylabel('Control'); plt.legend()

# (3) Objective value
plt.subplot(1,3,3)
plt.plot(loss)
plt.title('Objective Value'); plt.xlabel('Iteration'); plt.ylabel('Objective')

plt.tight_layout()
plt.savefig('HW5_Problem3.png', dpi=300, bbox_inches='tight')

