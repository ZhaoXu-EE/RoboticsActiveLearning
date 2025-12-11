import numpy as np
import matplotlib.pyplot as plt

def compute_signal_prob(robot_pos, candidate_sources):
    dx = np.abs(candidate_sources[:, 0] - robot_pos[0])
    dy = np.abs(candidate_sources[:, 1] - robot_pos[1])
    p = np.zeros_like(dx, dtype=float)

    p[(dy == 3) & (dx <= 3)] = 0.25
    p[(dy == 2) & (dx <= 2)] = 1/3
    p[(dy == 1) & (dx <= 1)] = 0.5
    p[(dy == 0) & (dx == 0)] = 1.0
    return p

GRID_SIZE = 25
all_positions = np.array([(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)])

np.random.seed(42)
source = all_positions[np.random.randint(len(all_positions))]
robot = np.array(all_positions[np.random.randint(len(all_positions))])

belief = np.ones(len(all_positions)) / len(all_positions)
robot_path = [robot.copy()]
beliefs = []
visited = set(tuple(robot))  # 跟踪访问过的位置

# 新增参数
EPSILON = 0.95 # 初始探索概率
DECAY_RATE = 0.97  # 探索概率衰减率

for t in range(100):
    # ε-greedy策略 ==============================
    if np.random.rand() < EPSILON:
        # 探索阶段：优先前往未访问区域
        candidates = []
        for action in ['up', 'down', 'left', 'right']:
            new_pos = robot.copy()
            if action == 'up' and robot[1] < GRID_SIZE-1: new_pos[1] += 1
            elif action == 'down' and robot[1] > 0: new_pos[1] -= 1
            elif action == 'left' and robot[0] > 0: new_pos[0] -= 1
            elif action == 'right' and robot[0] < GRID_SIZE-1: new_pos[0] += 1
            
            if tuple(new_pos) not in visited:
                candidates.append(new_pos)
        
        # 如果有未访问的邻近区域优先前往
        if candidates:
            robot = candidates[np.random.choice(len(candidates))]
        else:
            # 随机选择动作
            action = np.random.choice(['up', 'down', 'left', 'right'])
            if action == 'up' and robot[1] < GRID_SIZE-1: robot[1] += 1
            elif action == 'down' and robot[1] > 0: robot[1] -= 1
            elif action == 'left' and robot[0] > 0: robot[0] -= 1
            elif action == 'right' and robot[0] < GRID_SIZE-1: robot[0] += 1
    else:
        # 利用阶段：朝向最大信念区域移动
        goal = all_positions[np.argmax(belief)]
        dx = goal[0] - robot[0]
        dy = goal[1] - robot[1]
        
        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
        else:
            direction = 'up' if dy > 0 else 'down'
        
        # 执行移动
        if direction == 'up' and robot[1] < GRID_SIZE-1: robot[1] += 1
        elif direction == 'down' and robot[1] > 0: robot[1] -= 1
        elif direction == 'left' and robot[0] > 0: robot[0] -= 1
        elif direction == 'right' and robot[0] < GRID_SIZE-1: robot[0] += 1
    
    EPSILON *= DECAY_RATE  # 探索概率衰减
    visited.add(tuple(robot))
    robot_path.append(robot.copy())

    # 传感器测量与贝叶斯更新 ========================
    true_prob = compute_signal_prob(robot, np.array([source]))[0]
    reading = 1 if np.random.rand() < true_prob else 0

    likelihood = compute_signal_prob(robot, all_positions)
    belief *= (likelihood if reading else (1 - likelihood))
    belief /= belief.sum()

    beliefs.append(belief.copy())

# --- Step 4: Visualization --- #
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
steps_to_plot = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for idx, step in enumerate(steps_to_plot):
    ax = axes.flat[idx]
    belief_map = beliefs[step-1].reshape((GRID_SIZE, GRID_SIZE))
    
    # Plot belief map (no transpose, correct orientation)
    ax.imshow(belief_map, origin='lower', cmap='inferno',
              extent=[-0.5, 24.5, -0.5, 24.5], vmin=0, vmax=belief_map.max())
    
    # Plot trajectory (convert to numpy array for slicing)
    path = np.array(robot_path[:step+1])
    ax.plot(path[:, 0], path[:, 1], 'w-', linewidth=0.5)
    ax.plot(path[:, 0], path[:, 1], 'wo', markersize=2)
    
    # Annotate key points
    ax.scatter(*source, color='lime', marker='*', s=150, label='Source')
    ax.scatter(*robot_path[0], color='cyan', marker='s', s=80, label='Start')
    ax.scatter(*robot_path[step], color='red', marker='o', s=80, label='Current')
    
    ax.set_title(f"Step {step:03d}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(-0.5, 24.5)
    
    if idx == 0:
        ax.legend(loc='upper right', fontsize=6)

plt.tight_layout()
plt.savefig('exploration_results.png', dpi=150)
plt.show()
