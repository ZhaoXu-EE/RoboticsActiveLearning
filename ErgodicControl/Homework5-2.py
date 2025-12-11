"""
ME455 - HW5 - Problem 2 · Ergodic control for a 2-D second-order system
-----------------------------------------------------------------------
Compares to the teacher’s Figure 9:
  • left  : converged ergodic trajectory (with black dots + start marker)
  • centre: optimal acceleration controls u₁(t), u₂(t)
  • right : objective value over iLQR iterations (should drop fast)

The code re-uses the Fourier-basis metric & GMM target from Problem 1, but
extends the dynamics to a double integrator   ẍ = u   (state dimension 4).
Only the (x,y) position part is fed into the ergodic metric.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.stats import multivariate_normal

# ─────────────────────────── Target GMM ──────────────────────────────
w  = [0.5, 0.2, 0.3]
mu = [np.array([0.35, 0.38]), np.array([0.68, 0.25]), np.array([0.56, 0.64])]
S  = [np.array([[0.01,  0.004], [ 0.004, 0.01 ]]),
      np.array([[0.005,-0.003], [-0.003, 0.005]]),
      np.array([[0.008,  0.   ], [ 0.   , 0.004]])]

# Fourier basis (same 25×25 grid as P-1)
K       = 25
k_list  = np.array([[i, j] for i in range(K) for j in range(K)])
phi     = np.array([
    sum(wi * np.exp(-0.5 * (np.pi*k) @ Si @ (np.pi*k)) * np.cos(np.pi*k @ mi)
        for wi, mi, Si in zip(w, mu, S))
    for k in k_list
])
Lambda  = (1 + np.linalg.norm(k_list, axis=1)**2)**(-1.5)          # importance

# ─────────────────────────── Parameters ──────────────────────────────
dt, T   = 0.1, 10.0
N       = int(T/dt)
x0_pos  = np.array([0.30, 0.30])                                   # start [x,y]
x0      = np.hstack([x0_pos, np.zeros(2)])                         # [x y vx vy]

n_x     = 4         # full state dim  (x,y,vx,vy)
n_u     = 2         # control dim     (ax,ay)

Qz  = np.eye(n_x) * 1e-3            # quadratic on z(t)
Ru  = np.eye(n_u) * 2e-3            # control effort
Rv  = np.eye(n_u) * 1e-3            # descent-direction weight
BARR= 1e-1                          # soft box barrier for 0≤x,y≤1

# constant linearisation matrices for double integrator
A = np.zeros((n_x, n_x))
A[0,2] = A[1,3] = 1.0
B = np.zeros((n_x, n_u))
B[2,0] = B[3,1] = 1.0
invRv      = np.linalg.inv(Rv)
BRinvBT    = B @ invRv @ B.T

# ─────────────────────────── Helpers ─────────────────────────────────
def step(x, u):
    """One RK-Euler step of the double integrator."""
    xdot = np.array([x[2], x[3], u[0], u[1]])
    return x + dt * xdot

def simulate(x0, U):
    X = np.zeros((N, n_x))
    x = x0.copy()
    for t in range(N):
        X[t] = x
        x    = step(x, U[t])
    return X

def compute_a_k(X):
    pos = X[:, :2]                           # only (x,y) matter
    return np.cos(np.pi * pos @ k_list.T).sum(0) * dt / T

def grad_state(x, ck):
    """∂E/∂x — gradient w.r.t. full state (pos grad, zero on velocities)."""
    pos = x[:2]
    s   = -2 * Lambda * (ck - phi) * np.sin(np.pi * k_list @ pos)
    grad_pos = (np.pi * dt / T) * (s[:,None] * k_list).sum(0)

    # soft barrier on workspace [0,1]²
    barr = np.where(pos < 0, 2*pos, 0.) + np.where(pos > 1, 2*(pos-1), 0.)
    return np.hstack([grad_pos + BARR*barr, np.zeros(2)])

# ─────────────────────────── One iLQR iteration ─────────────────────
def ilqr_iter(U):
    X  = simulate(x0, U)
    ck = compute_a_k(X)

    a_list = np.zeros((N, n_x))
    b_list = np.zeros((N, n_u))
    for i in range(N):
        a_list[i] = grad_state(X[i], ck)     # ∂E/∂x
        b_list[i] = 2 * Ru @ U[i]            # ∂E/∂u

    # ODE for [z; p]  (size 2·n_x).  Piece-wise constant coefficients.
    def zp_dyn(t_vec, zp_mat):
        out = np.empty_like(zp_mat)
        idx = np.minimum((t_vec/dt).astype(int), N-1)
        for col, j in enumerate(idx):
            M11, M22 = A, -A.T
            M12      = -BRinvBT
            M21      = -Qz
            M        = np.block([[M11, M12],
                                 [M21, M22]])
            v1 = -B @ invRv @ b_list[j]
            v2 = -a_list[j]
            v  = np.hstack([v1, v2])
            out[:,col] = M @ zp_mat[:,col] + v
        return out

    def bc(z0, zT):          # z(0)=0 , p(T)=0
        return np.hstack([z0[:n_x], zT[n_x:]])

    grid = np.linspace(0, (N-1)*dt, N)
    sol  = solve_bvp(zp_dyn, bc, grid, np.zeros((2*n_x, N)), max_nodes=5*N)
    p_traj = sol.sol(grid)[n_x:].T                         # (N , n_x)

    # descent direction   v(t) = argmin ½vᵀR_v v + pᵀBv + bᵀv
    V = -(invRv @ (B.T @ p_traj.T + b_list.T)).T          # (N , 2)
    return V, ck, X

# ─────────────────────────── Optimisation loop ──────────────────────
rng   = np.random.default_rng(0)
U     = rng.normal(0, 0.3, size=(N, n_u))      # random accel. seed
loss  = []
alpha, beta = 1e-3, 0.8                        # Armijo
for it in range(120):                          # ~ 20-40 iterations usually enough
    V, ck, X = ilqr_iter(U)
    J        = (Lambda * (ck-phi)**2).sum()
    loss.append(J)

    g = 1.0                                    # line search
    while True:
        U_new = U + g * V
        J_new = (Lambda * (compute_a_k(simulate(x0, U_new))-phi)**2).sum()
        if J_new <= J - alpha*g*(V**2).sum():
            break
        g *= beta
        if g < 1e-6:  break
    U = U_new

# ─────────────────────────── Plot & save ────────────────────────────
t = np.linspace(0, T, N)
X = simulate(x0, U)
pos = X[:, :2]

plt.figure(figsize=(15,4))

# (1) Trajectory & target density
plt.subplot(1,3,1)
xx, yy = np.meshgrid(np.linspace(0,1,200), np.linspace(0,1,200))
Z = sum(wi*multivariate_normal(mi,Si).pdf(np.dstack((xx,yy)))
        for wi,mi,Si in zip(w,mu,S))
plt.contourf(xx,yy,Z,20,cmap='Reds',alpha=0.7)
plt.plot(pos[:,0], pos[:,1], 'k-', lw=2)
plt.scatter(pos[:,0], pos[:,1], c='k', s=6, zorder=2)     # little dots
plt.scatter(x0_pos[0], x0_pos[1], c='dodgerblue', s=80, zorder=3)
plt.title('Ergodic Trajectory');  plt.axis('equal')
plt.xlim(0,1); plt.ylim(0,1); plt.xlabel('X'); plt.ylabel('Y')

# (2) Optimal controls  (smoothed for readability only)
def smooth(v): return np.convolve(v, np.ones(3)/3, mode='same')
plt.subplot(1,3,2)
plt.plot(t, smooth(U[:,0]), label=r'$u_1(t)$')
plt.plot(t, smooth(U[:,1]), label=r'$u_2(t)$')
plt.title('Optimal Control');  plt.xlabel('Time'); plt.ylabel('Control'); plt.legend()

# (3) Objective value over iLQR iterations
plt.subplot(1,3,3)
plt.plot(loss); plt.title('Objective Value')
plt.xlabel('Iteration'); plt.ylabel('Objective')

plt.tight_layout()
plt.savefig('HW5_Problem2.png', dpi=300, bbox_inches='tight')
print(f'Done!  Final objective ≈ {loss[-1]:.4e}  → see “HW5_Problem2.png”.')
