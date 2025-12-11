import numpy as np
import matplotlib.pyplot as plt
from math import log

def compute_signal_prob(robot_pos, candidate_sources):
    dx = np.abs(candidate_sources[:, 0] - robot_pos[0])
    dy = np.abs(candidate_sources[:, 1] - robot_pos[1])
    p = np.zeros_like(dx, dtype=float)
    
    p[(dy == 3) & (dx <= 3)] = 0.25
    p[(dy == 2) & (dx <= 2)] = 1/3
    p[(dy == 1) & (dx <= 1)] = 0.5
    p[(dy == 0) & (dx == 0)] = 1.0
    return p

def entropy(belief):
    log_b = np.where(belief > 1e-10, np.log(belief), 0)
    return -np.sum(belief * log_b)

def infotaxis_step(robot, belief, all_positions):
    actions = []
    directions = []
    
    # Generate valid actions
    if robot[1] < GRID_SIZE-1:
        actions.append(robot + [0, 1])
        directions.append('up')
    if robot[1] > 0:
        actions.append(robot + [0, -1])
        directions.append('down')
    if robot[0] > 0:
        actions.append(robot + [-1, 0])
        directions.append('left')
    if robot[0] < GRID_SIZE-1:
        actions.append(robot + [1, 0])
        directions.append('right')
    
    current_entropy = entropy(belief)
    best_gain = -np.inf
    best_action = robot
    
    for action_pos, direction in zip(actions, directions):
        # Predict measurement probabilities
        p_signal = compute_signal_prob(action_pos, all_positions)
        
        # Calculate expected entropy for both possible measurements
        # Case 1: measurement=1
        belief_1 = belief * p_signal
        belief_1 /= belief_1.sum()
        entropy_1 = entropy(belief_1)
        
        # Case 2: measurement=0
        belief_0 = belief * (1 - p_signal)
        belief_0 /= belief_0.sum()
        entropy_0 = entropy(belief_0)
        
        # Expected entropy after action
        p_measure1 = np.sum(belief * p_signal)
        expected_entropy = p_measure1*entropy_1 + (1-p_measure1)*entropy_0
        
        # Information gain
        gain = current_entropy - expected_entropy
        
        if gain > best_gain:
            best_gain = gain
            best_action = action_pos
    
    return best_action, best_gain

def run_trial(trial_num):
    np.random.seed(42 + trial_num)
    source = all_positions[np.random.randint(len(all_positions))]
    robot = np.array(all_positions[np.random.randint(len(all_positions))])
    
    belief = np.ones(len(all_positions)) / len(all_positions)
    robot_path = [robot.copy()]
    beliefs = []
    entropy_history = []
    
    step = 0
    converged = False
    while not converged and step < 500:  # 最大步数限制
        # 选择动作
        new_robot, gain = infotaxis_step(robot, belief, all_positions)
        
        # 执行移动
        robot = new_robot.copy()
        robot_path.append(robot.copy())
        
        # 模拟传感器测量
        true_prob = compute_signal_prob(robot, np.array([source]))[0]
        reading = 1 if np.random.rand() < true_prob else 0
        
        # 贝叶斯更新
        likelihood = compute_signal_prob(robot, all_positions)
        belief *= (likelihood if reading else (1 - likelihood))
        belief /= belief.sum()
        
        beliefs.append(belief.copy())
        entropy_history.append(entropy(belief))
        
        # 检查收敛条件
        if np.max(belief) > 0.99 or (len(entropy_history) > 10 and 
                                   np.std(entropy_history[-10:]) < 0.01):
            converged = True
        
        step += 1
    
    # 选择可视化时间点
    selected_steps = []
    target_counts = 10
    if step < target_counts:
        selected_steps = list(range(1, step+1))
    else:
        interval = max(1, step // target_counts)
        selected_steps = list(range(0, step, interval))[:target_counts]
        selected_steps[-1] = step-1  # 确保包含最后一步
    
    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    for idx, t in enumerate(selected_steps):
        ax = axes.flat[idx]
        if t >= len(beliefs):
            continue
            
        belief_map = beliefs[t].reshape((GRID_SIZE, GRID_SIZE))
        ax.imshow(belief_map, origin='lower', cmap='inferno',
                 extent=[-0.5, 24.5, -0.5, 24.5], vmin=0, vmax=belief_map.max())
        
        path = np.array(robot_path[:t+2])
        ax.plot(path[:, 0], path[:, 1], 'w-', linewidth=1)
        ax.scatter(*source, color='lime', marker='*', s=200, zorder=3)
        ax.scatter(*robot_path[0], color='cyan', marker='s', s=100)
        ax.scatter(*robot_path[t+1], color='red', marker='o', s=100)
        ax.set_title(f"Step {t+1}\nEntropy: {entropy_history[t]:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(f"Infotaxis Trial {trial_num+1} (Converged at step {step})", y=1.02)
    plt.tight_layout()
    plt.savefig(f'infotaxis_trial_{trial_num+1}.png', bbox_inches='tight')
    plt.close()

# 参数设置
GRID_SIZE = 25
all_positions = np.array([(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)])

# 运行3次试验
for trial in range(3):
    run_trial(trial)