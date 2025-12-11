"""
ME455 - HW5 - Problem 1 · 2-D first-order ergodic control
(2025-05-30 smooth revision)
----------------------------------------------------------
主要修改点
1. **控制成本加大**        Ru = 2 × 10⁻²   ← 抑制高频抖动
2. **边界势垒加重**        BARR = 0.5      ← 减少越界导致的折点
3. **随机初始控制更小**    σ = 0.10        ← 避免一开始就“乱跑”
4. **线搜索更谨慎**        beta = 0.6      ← 减少过大步长带来的折返
5. 仅绘图时继续做 3-点滑窗平滑，算法本身不改动
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.stats import multivariate_normal

# ───────────────────────── 目标高斯混合 ─────────────────────────
w  = [0.5, 0.2, 0.3]
mu = [np.array([0.35, 0.38]),
      np.array([0.68, 0.25]),
      np.array([0.56, 0.64])]
S  = [np.array([[0.01 ,  0.004],
                [0.004,  0.01 ]]),
      np.array([[0.005, -0.003],
                [-0.003, 0.005]]),
      np.array([[0.008,  0.    ],
                [0.    , 0.004]])]

# ───────── Fourier 基 & 目标频谱 φ_k  (25×25 = 625 个) ──────────
K       = 25
k_list  = np.array([[i, j] for i in range(K) for j in range(K)])
num_k   = len(k_list)

def phi_k(k):
    kπ = np.pi * k
    return sum(wi * np.exp(-0.5 * kπ @ Si @ kπ) * np.cos(kπ @ mi)
               for wi, mi, Si in zip(w, mu, S))

phi     = np.array([phi_k(k) for k in k_list])
Lambda  = (1 + np.linalg.norm(k_list, axis=1)**2)**(-1.5)

# ─────────────────────────── 参数设置 ───────────────────────────
dt, T = 0.1, 10.0
N     = int(T/dt)
x0    = np.array([0.30, 0.30])
n     = 2                          # 状态 / 控制 维度

Qz   = np.eye(n) * 1e-3
Ru   = np.eye(n) * 2e-2            # ←↑ 由 2e-3 提高 10×
Rv   = np.eye(n) * 1e-3            # 下降方向正则
BARR = 0.5                         # ←↑ 边界势垒

# ─────────────────────────── 动力学 ────────────────────────────
def step(x, u):          # Euler 积分：ẋ = u
    return x + dt * u

def simulate(x0, U):
    X, x = np.zeros((N, n)), x0.copy()
    for t in range(N):
        X[t] = x
        x    = step(x, U[t])
    return X

# ──────────────────────── E-metric 构件 ────────────────────────
def compute_a_k(X):
    return np.cos(np.pi * X @ k_list.T).sum(0) * dt / T

def grad_x(x, ck):
    s = -2 * Lambda * (ck - phi) * np.sin(np.pi * k_list @ x)
    return (np.pi * dt / T) * (s[:, None] * k_list).sum(0)

def barr_grad(x):        # soft-barrier ∝ 距离边界的平方
    return np.where(x < 0, 2*x, 0.) + np.where(x > 1, 2*(x-1), 0.)

# ───────────────────────── iLQR 单次迭代 ───────────────────────
def ilqr_iter(U):
    X  = simulate(x0, U)
    ck = compute_a_k(X)

    a_list = np.zeros((N, n))      # ∂E/∂x
    b_list = np.zeros((N, n))      # ∂E/∂u
    for i in range(N):
        a_list[i] = grad_x(X[i], ck) + BARR * barr_grad(X[i])
        b_list[i] = 2 * Ru @ U[i]

    invRv = np.linalg.inv(Rv)

    def dyn(t_vec, zp_mat):
        out = np.empty_like(zp_mat)
        idx = np.minimum((t_vec/dt).astype(int), N-1)
        for col, j in enumerate(idx):
            a, b = a_list[j], b_list[j]
            M = np.block([[np.zeros((n, n)), -invRv],
                          [-Qz            ,  np.zeros((n, n))]])
            v = np.hstack([-invRv @ b, -a])
            out[:, col] = M @ zp_mat[:, col] + v
        return out

    def bc(z0, zT):               # p(T)=0, z(0)=0
        return np.hstack([z0[:n], zT[n:]])

    grid = np.linspace(0, (N-1)*dt, N)
    sol  = solve_bvp(dyn, bc, grid,
                     np.zeros((2*n, N)), max_nodes=5*N)
    p_traj = sol.sol(grid)[n:].T               # 共轭变量 p(t)

    V = -(invRv @ (p_traj + b_list).T).T       # 下降方向
    return V, ck, X

# ────────────────────────── 主循环 ─────────────────────────────
rng   = np.random.default_rng(0)
U     = rng.normal(0, 0.10, size=(N, n))       # ← σ 从 0.4 ↓ 0.10
loss  = []
alpha, beta = 1e-3, 0.6                        # ← β = 0.6 更保守

for _ in range(160):
    V, ck, X = ilqr_iter(U)
    J = (Lambda * (ck - phi)**2).sum()
    loss.append(J)

    # Armijo line-search
    g = 1.0
    while True:
        U_new = U + g * V
        J_new = (Lambda *
                 (compute_a_k(simulate(x0, U_new)) - phi)**2).sum()
        if J_new <= J - alpha * g * (V**2).sum():
            break
        g *= beta
        if g < 1e-6:
            break
    U = U_new

# ─────────────────────────── 绘   图 ───────────────────────────
t  = np.linspace(0, T, N)
Xf = simulate(x0, U)

plt.figure(figsize=(15, 4))

# (1) Trajectory
plt.subplot(1, 3, 1)
xx, yy = np.meshgrid(np.linspace(0, 1, 200),
                     np.linspace(0, 1, 200))
pos = np.dstack((xx, yy))
Z   = sum(wi *
          multivariate_normal(mi, Si).pdf(pos)
          for wi, mi, Si in zip(w, mu, S))
plt.contourf(xx, yy, Z, 20, cmap='Reds', alpha=0.7)
plt.plot(Xf[:, 0], Xf[:, 1], 'k-', lw=2)
plt.scatter(Xf[:, 0], Xf[:, 1], c='k', s=6, zorder=2)
plt.scatter(*x0, c='dodgerblue', s=80, zorder=3)
plt.title('Ergodic Trajectory')
plt.xlabel('X'); plt.ylabel('Y')
plt.axis('equal'); plt.xlim(0, 1); plt.ylim(0, 1)

# (2) Control   —— 3-点滑窗仅用于可视化
def smooth(v): return np.convolve(v, np.ones(3)/3, mode='same')
plt.subplot(1, 3, 2)
plt.plot(t, smooth(U[:, 0]), label=r'$u_x(t)$')
plt.plot(t, smooth(U[:, 1]), label=r'$u_y(t)$')
plt.title('Optimal Control')
plt.xlabel('Time'); plt.ylabel('Control')
plt.legend()

# (3) Objective
plt.subplot(1, 3, 3)
plt.plot(loss)
plt.title('Objective Value')
plt.xlabel('Iteration'); plt.ylabel('Objective')

plt.tight_layout()
plt.savefig('HW5_Problem1.png', dpi=300, bbox_inches='tight')
print(f'Finished.  Final objective = {loss[-1]:.3e}')
