"""
Infotaxis search agent for MurpheyLab ME455 box_gym_project
-----------------------------------------------------------
Author : Xu Zhao  (2025-06-04)
Purpose: Demo for the final project – autonomous “box finder”
Key idea:
    • Maintain a Bayesian belief b(i) over each box i being “interesting”.
    • For every legal arm motion a, predict the expected entropy H(after a).
    • Execute the action that maximises current-entropy − expected-entropy.
    • Update belief with the binary measurement returned by env.step().
Works well for small-to-mid sized grids (<50 boxes); for larger grids you
may accelerate with vectorised NumPy/JAX.
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────
# 0.  Environment
# ────────────────────────────────────────────────────────────────────
ENV_NAME = "BoxSearch-v0"          # 在 test_box_gym.ipynb 里看过的注册名
env = gym.make(ENV_NAME)

obs = env.reset(seed=0)            # obs 里通常有 {robot_state, box_id, reading}
n_boxes = env.unwrapped.n_boxes    # 或者 len(env.box_positions)

# Likelihood table  P(z=1 | robot_pose, box_id)  —— 这里直接用环境给的接口;
# 如果需要自己写，参见作业 HW1/HW2 的 measurement model.
def likelihood(reading_one: bool, robot_pose, box_id):
    """
    Wrap env's internal generative model so we can query both outcomes.
    Faster than calling env.predict repeatedly if you vectorise.
    """
    p1 = env.unwrapped.prob_positive(robot_pose, box_id)
    return p1 if reading_one else (1.0 - p1)

# ────────────────────────────────────────────────────────────────────
# 1.  Belief & entropy helpers
# ────────────────────────────────────────────────────────────────────
belief = np.ones(n_boxes) / n_boxes

def entropy(b):
    """Shannon entropy (base-e)."""
    logb = np.where(b > 1e-12, np.log(b), 0.)
    return -np.sum(b * logb)

def expected_entropy_after(action, robot_pose):
    """Compute E[H(b’)] if we move with 'action' and then observe."""
    next_pose = env.unwrapped.predict_pose(robot_pose, action)
    # Vectorised likelihoods for all boxes
    p1 = np.array([env.unwrapped.prob_positive(next_pose, i)
                   for i in range(n_boxes)])

    # Belief update for z=1
    b1 = belief * p1
    if b1.sum() > 0:
        b1 /= b1.sum()
    H1 = entropy(b1)

    # Belief update for z=0
    b0 = belief * (1 - p1)
    if b0.sum() > 0:
        b0 /= b0.sum()
    H0 = entropy(b0)

    p_measure1 = np.sum(belief * p1)      # P(z=1)
    return p_measure1 * H1 + (1 - p_measure1) * H0

# ────────────────────────────────────────────────────────────────────
# 2.  Main loop
# ────────────────────────────────────────────────────────────────────
max_steps = 300
robot_pose = env.unwrapped.robot_pose
traj, ent_hist = [], [entropy(belief)]

for step in range(max_steps):
    traj.append(robot_pose.copy())

    # 2-A. enumerate legal actions from current pose
    legal_actions = env.unwrapped.legal_actions(robot_pose)

    # 2-B. pick the one with best information gain
    H_curr = ent_hist[-1]
    gains = []
    for a in legal_actions:
        H_exp = expected_entropy_after(a, robot_pose)
        gains.append(H_curr - H_exp)
    best_a = legal_actions[int(np.argmax(gains))]

    # 2-C. execute & observe
    obs, reward, done, info = env.step(best_a)
    z = obs["signal"]          # binary measurement 0/1
    box_idx = obs["box_id"]
    robot_pose = obs["robot_pose"]

    # 2-D. Bayesian update  (only one box sensed — faster)
    p_z = env.unwrapped.prob_positive(robot_pose, box_idx)
    if z == 1:
        belief[box_idx] *= p_z
    else:
        belief[box_idx] *= (1 - p_z)
    belief /= belief.sum()
    ent_hist.append(entropy(belief))

    # 2-E. termination conditions
    if belief.max() > 0.98 or done:
        print(f"[✓] Converged at step {step+1}")
        break

# ────────────────────────────────────────────────────────────────────
# 3.  Quick-look plot (optional)
# ────────────────────────────────────────────────────────────────────
plt.figure(figsize=(6,4))
plt.plot(ent_hist, label="Belief entropy")
plt.xlabel("Timestep"); plt.ylabel("Entropy (nats)")
plt.title("Infotaxis entropy reduction")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()
