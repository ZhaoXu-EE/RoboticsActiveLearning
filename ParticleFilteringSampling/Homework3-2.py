"""
ME455 – Homework 3  Problem 2
Differential-drive robot + particle filter
Author: <your-name>
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

# ------------------------------------------------------------------
# 0.  Problem parameters (from the PDF)
# ------------------------------------------------------------------
DT                = 0.05           # integration step
T_FINAL           = 5.0            # seconds
N_STEPS           = int(T_FINAL / DT)          # 100
CTRL_MEAN         = np.array([0.5, -0.63])     # [u1, u2]
CTRL_NOISE_COV    = np.diag([0.04, 0.02])      # process noise on control
MEAS_NOISE_COV    = np.diag([0.004, 0.004, 0.002])
INIT_STATE        = np.array([0.0, 0.0, np.pi/2])
INIT_COV          = np.diag([1e-3, 1e-3, 1e-4])

N_PARTICLES       = 1_000
SNAPSHOTS         = [0, 20, 40, 60, 80, 100]    # time indices to plot

rng = np.random.default_rng(seed=2025)          # reproducibility

# ------------------------------------------------------------------
# 1.  Dynamics – 4th-order Runge-Kutta
# ------------------------------------------------------------------
def f(state, ctrl):
    """Differential-drive dynamics  ṡ = f(s,u).  state=[x,y,θ]."""
    x, y, th = state.T
    u1, u2   = ctrl.T
    dx       = np.cos(th) * u1
    dy       = np.sin(th) * u1
    dth      = u2
    return np.vstack([dx, dy, dth]).T

def rk4_step(state, ctrl, dt=DT):
    k1 = f(state,               ctrl)
    k2 = f(state + 0.5*dt*k1,   ctrl)
    k3 = f(state + 0.5*dt*k2,   ctrl)
    k4 = f(state +     dt*k3,   ctrl)
    return state + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)

# ------------------------------------------------------------------
# 2.  Utilities
# ------------------------------------------------------------------
def propagate(state, noisy_ctrl):
    """One forward step for *one* state vector."""
    return rk4_step(state[None, :], noisy_ctrl[None, :])[0]

def measurement(gt_state):
    """Simulate noisy measurement  z ~ N(gt, R)."""
    return gt_state + rng.multivariate_normal(np.zeros(3), MEAS_NOISE_COV)

def gaussian_likelihood(sr, mean, cov_inv, log_det):
    """Log-likelihood of sr under N(mean, cov)."""
    diff = sr - mean
    return -0.5 * (np.einsum('...i,ij,...j', diff, cov_inv, diff) + log_det + 3*np.log(2*np.pi))

R_inv     = np.linalg.inv(MEAS_NOISE_COV)
log_det_R = np.log(np.linalg.det(MEAS_NOISE_COV))

def resample(states, weights):
    idx = rng.choice(len(states), size=len(states), replace=True, p=weights)
    states[:]  = states[idx]
    weights[:] = 1.0 / len(states)
    return states, weights

# ------------------------------------------------------------------
# 3.  Ground-truth simulation (with process noise)
# ------------------------------------------------------------------
gt_states  = np.empty((N_STEPS+1, 3))
gt_states[0] = INIT_STATE.copy()

for t in range(N_STEPS):
    noisy_u  = CTRL_MEAN + rng.multivariate_normal(np.zeros(2), CTRL_NOISE_COV)
    gt_states[t+1] = propagate(gt_states[t], noisy_u)

# ------------------------------------------------------------------
# 4-A.  Particle Filter – *prediction only*
# ------------------------------------------------------------------
particles_pred = rng.multivariate_normal(INIT_STATE, INIT_COV, size=N_PARTICLES)
weights_pred   = np.full(N_PARTICLES, 1.0/N_PARTICLES)

snap_pred = []      # store particles at snapshot times
if 0 in SNAPSHOTS:                               # initial state
    snap_pred.append(particles_pred.copy())      # for t=0

for t in range(N_STEPS):
    # sample the *same* noise model as we think governs the process
    noisy_u_particles = CTRL_MEAN + rng.multivariate_normal(np.zeros(2), CTRL_NOISE_COV, size=N_PARTICLES)
    particles_pred = rk4_step(particles_pred, noisy_u_particles)
    if (t+1) in SNAPSHOTS:
        snap_pred.append(particles_pred.copy())

# ------------------------------------------------------------------
# 4-B.  Particle Filter – *with measurement update + resampling*
# ------------------------------------------------------------------
particles_upd = rng.multivariate_normal(INIT_STATE, INIT_COV, size=N_PARTICLES)
weights_upd   = np.full(N_PARTICLES, 1.0/N_PARTICLES)

snap_upd = []
if 0 in SNAPSHOTS:                               # initial state
    snap_upd.append(particles_upd.copy())        # for t=0

for t in range(N_STEPS):
    # prediction
    noisy_u_particles = CTRL_MEAN + rng.multivariate_normal(np.zeros(2), CTRL_NOISE_COV, size=N_PARTICLES)
    particles_upd = rk4_step(particles_upd, noisy_u_particles)

    # measurement at time t+1  (robot is at gt_states[t+1])
    z = measurement(gt_states[t+1])

    # importance weights ∝ p(z | s_i)
    log_w = gaussian_likelihood(particles_upd, z, R_inv, log_det_R)
    log_w -= log_w.max()         # numerical stability
    weights_upd = np.exp(log_w)
    weights_upd /= weights_upd.sum()

    # systematic multinomial resampling  (every step, for simplicity)
    particles_upd, weights_upd = resample(particles_upd, weights_upd)

    if (t+1) in SNAPSHOTS:
        snap_upd.append(particles_upd.copy())

# ------------------------------------------------------------------
# 5.  Visualisation
# ------------------------------------------------------------------
def make_figure(snapshots, title):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for k, step in enumerate(SNAPSHOTS):
        ax = axes[k]
        ax.scatter(snapshots[k][:, 0], snapshots[k][:, 1], s=4, alpha=0.5, label='particles')
        ax.plot(gt_states[step, 0], gt_states[step, 1], 'r*', ms=12, label='ground truth')
        ax.set_title(f't={step*DT:.1f}s')
        ax.set_xlim(-1, 2); ax.set_ylim(-0.5, 2)
        ax.set_aspect('equal')
    axes[0].legend(loc='upper right')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# --
# 6.  Visualisation – 1D projection
# --
def one_axis_figure(snap_pred, snap_upd):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    titles = ['Particle Filter (Prediction Only)', 'Particle Filter (Resampling)']

    for col, (snapshots, ax) in enumerate(zip([snap_pred, snap_upd], axes)):
        # 叠画 6 个时间点
        for k, step in enumerate(SNAPSHOTS):
            pts = snapshots[k]
            ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.1, color='tab:blue')
            ax.scatter(gt_states[step, 0], gt_states[step, 1],
                       s=100, marker='o', color='tab:orange',
                       edgecolor='k', label='Ground Truth' if k == 0 else '')
            # 粒子均值
            # mu = pts.mean(0)
            # ax.scatter(mu[0], mu[1], marker='v', color='tab:orange', s=40)

            # 取地面真值朝向（θ），画一个朝向 θ 的等边三角
            x0, y0, th = gt_states[step]
            # tri = RegularPolygon((x0, y0), numVertices=3, radius=0.07,
            #                     orientation=th,  facecolor='tab:orange', edgecolor='k')
            dx, dy = 0.12 * np.cos(th), 0.12 * np.sin(th)   # 0.12 ≈ 箭头长度
            ax.quiver(x0, y0, dx, dy,
                    angles='xy', scale_units='xy', scale=1,
                    width=0.006, color='tab:orange', edgecolor='k', zorder=3)
            # ax.add_patch(tri)


        ax.set_title(titles[col])
        ax.set_xlim(-0.2, 1.8)
        ax.set_ylim(-0.3, 1.1)
        ax.set_aspect('equal')
        ax.legend(loc='upper left')

    plt.tight_layout()
    # plt.show()
    plt.savefig('pf_1d_projection.png', dpi=300)
    plt.close(fig)


# fig1 = make_figure(snap_pred, 'Particle Filter – prediction only')
# fig2 = make_figure(snap_upd, 'Particle Filter – with resampling')

# plt.show()
# fig1.savefig('pf_pred.png', dpi=300)
# fig2.savefig('pf_upd.png', dpi=300)
# plt.close(fig1)
# plt.close(fig2)


one_axis_figure(snap_pred, snap_upd)