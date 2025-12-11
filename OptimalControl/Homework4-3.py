import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import pdb

def dyn(xt, ut):
    """Dynamics function for differential drive vehicle."""
    xdot = np.array([
        np.cos(xt[2]) * ut[0],
        np.sin(xt[2]) * ut[0],
        ut[1]
    ])
    return xdot

def get_A(t, xt, ut):
    """Linearized dynamics with respect to state."""
    A = np.array([
        [0.0, 0.0, -np.sin(xt[2]) * ut[0]],
        [0.0, 0.0, np.cos(xt[2]) * ut[0]],
        [0.0, 0.0, 0.0]
    ])
    return A

def get_B(t, xt, ut):
    """Linearized dynamics with respect to control."""
    B = np.array([
        [np.cos(xt[2]), 0.0],
        [np.sin(xt[2]), 0.0],
        [0.0, 1.0]
    ])
    return B

def step(xt, ut):
    """Integrate dynamics using RK4."""
    dt = 0.1  # Time step
    k1 = dt * dyn(xt, ut)
    k2 = dt * dyn(xt + 0.5 * k1, ut)
    k3 = dt * dyn(xt + 0.5 * k2, ut)
    k4 = dt * dyn(xt + k3, ut)
    xt_new = xt + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return xt_new

def traj_sim(x0, ulist):
    """Simulate trajectory given initial state and control sequence."""
    tsteps = ulist.shape[0]
    x_traj = np.zeros((tsteps, 3))
    xt = x0.copy()
    for t in range(tsteps):
        xt_new = step(xt, ulist[t])
        x_traj[t] = xt_new.copy()
        xt = xt_new.copy()
    return x_traj

def loss(t, xt, ut):
    """Loss function for tracking."""
    xd = np.array([
        2.0*t / np.pi, 0.0, np.pi/2.0
    ])  # desired system state at time t

    x_loss = (xt - xd).T @ Q_x @ (xt - xd)
    u_loss = ut.T @ R_u @ ut

    return x_loss + u_loss

def dldx(t, xt, ut):
    """Derivative of loss with respect to state."""
    xd = np.array([
        2.0*t / np.pi, 0.0, np.pi/2.0
    ])

    dvec = 2 * Q_x @ (xt - xd)
    return dvec

def dldu(t, xt, ut):
    """Derivative of loss with respect to control."""
    dvec = 2 * R_u @ ut
    return dvec

def ilqr_iter(x0, u_traj):
    """
    One iteration of iLQR algorithm.
    Returns: the descent direction v_traj
    """
    # Forward simulate the state trajectory
    x_traj = traj_sim(x0, u_traj)

    # Compute matrices needed for specifying the dynamics of z(t) and p(t)
    A_list = np.zeros((tsteps, 3, 3))
    B_list = np.zeros((tsteps, 3, 2))
    a_list = np.zeros((tsteps, 3))
    b_list = np.zeros((tsteps, 2))
    for t_idx in range(tsteps):
        t = t_idx * dt
        A_list[t_idx] = get_A(t, x_traj[t_idx], u_traj[t_idx])
        B_list[t_idx] = get_B(t, x_traj[t_idx], u_traj[t_idx])
        a_list[t_idx] = dldx(t, x_traj[t_idx], u_traj[t_idx])
        b_list[t_idx] = dldu(t, x_traj[t_idx], u_traj[t_idx])

    xd_T = np.array([
        2.0*(tsteps-1)*dt / np.pi, 0.0, np.pi/2.0
    ])  # desired terminal state
    p1 = 2 * P1 @ (x_traj[-1] - xd_T)

    def zp_dyn(t, zp):
        t_idx = int(t/dt)
        At = A_list[t_idx]
        Bt = B_list[t_idx]
        at = a_list[t_idx]
        bt = b_list[t_idx]

        M_11 = At
        M_12 = -Bt @ np.linalg.inv(R_v) @ Bt.T
        M_21 = -Q_z
        M_22 = -At.T
        dyn_mat = np.block([
            [M_11, M_12],
            [M_21, M_22]
        ])

        m_1 = -Bt @ np.linalg.inv(R_v) @ bt
        m_2 = -at
        dyn_vec = np.hstack([m_1, m_2])

        return dyn_mat @ zp + dyn_vec

    # Convert function for solve_bvp format
    def zp_dyn_list(t_list, zp_list):
        list_len = len(t_list)
        zp_dot_list = np.zeros((6, list_len))
        for _i in range(list_len):
            zp_dot_list[:,_i] = zp_dyn(t_list[_i], zp_list[:,_i])
        return zp_dot_list

    # Boundary condition
    def zp_bc(zp_0, zp_T):
        return np.array([zp_0[:3], zp_T[3:] - p1]).flatten()

    # Solve the boundary value problem
    tlist = np.arange(tsteps) * dt
    res = solve_bvp(
        zp_dyn_list, zp_bc, tlist, np.zeros((6,tsteps)),
        max_nodes=100
    )
    zp_traj = res.sol(tlist).T

    z_traj = zp_traj[:,:3]
    p_traj = zp_traj[:,3:]

    # Calculate the descent direction
    v_traj = np.zeros((tsteps, 2))
    for _i in range(tsteps):
        Bt = B_list[_i]
        bt = b_list[_i]
        pt = p_traj[_i]
        vt = -np.linalg.inv(R_v) @ (Bt.T @ pt + bt)
        v_traj[_i] = vt

    return v_traj

# Define parameters
dt = 0.1
x0 = np.array([0.0, 0.0, np.pi/2.0])
tsteps = 63
init_u_traj = np.tile(np.array([1.0, -0.5]), reps=(tsteps,1))

# Cost function parameters
Q_x = np.diag([95.0, 10.0, 2.0])  # State cost
R_u = np.diag([4.0, 2.0])         # Control cost
P1 = np.diag([20.0, 20.0, 5.0])   # Terminal state cost

# Parameters for descent direction computation
Q_z = np.diag([5.0, 5.0, 1.0])    # Quadratic cost on z
R_v = np.diag([2.0, 1.0])         # Quadratic cost on v

# Start iLQR iterations
u_traj = init_u_traj.copy()
x_traj_initial = None
loss_list = []
time = np.arange(tsteps) * dt

# 完整修复解决方案 - 解决参数集2和3的初始轨迹不显示问题

# Main loop for three different parameter sets
for param_set in range(3):
    if param_set == 0:
        # Default parameters (already set above)
        u_traj = init_u_traj.copy()
        title = "Parameter Set 1 (Default)"
    elif param_set == 1:
        # Parameter set 2: Different initial control
        u_traj = np.tile(np.array([0.5, 0.2]), reps=(tsteps,1))
        title = "Parameter Set 2 (Different Initial Control)"
    elif param_set == 2:
        # Parameter set 3: Different cost parameters
        Q_x = np.diag([50.0, 20.0, 5.0])
        R_u = np.diag([2.0, 1.0])
        P1 = np.diag([30.0, 30.0, 10.0])
        u_traj = np.tile(np.array([0.8, 0.0]), reps=(tsteps,1)) 
        title = "Parameter Set 3 (Different Cost Weights)"
    
    # 重置变量
    loss_list = []
    
    # 重要：计算并保存初始轨迹
    initial_control = u_traj.copy()
    
    # 为了确保初始轨迹正确，我们对初始状态应用traj_sim函数
    # 注意：我们需要从x0开始而不是从全零数组开始
    x_traj_initial = np.zeros((tsteps, 3))
    x_traj_initial[0] = x0.copy()  # 设置初始状态
    
    # 手动模拟轨迹，确保它包含足够多的点
    xt = x0.copy()
    for t in range(tsteps-1):  # 注意：我们已经设置了第一个点，所以只需模拟剩余的点
        xt_new = step(xt, initial_control[t])
        x_traj_initial[t+1] = xt_new.copy()
        xt = xt_new.copy()
    
    # 打印调试信息以检查初始轨迹
    print(f"参数集 {param_set+1} 初始轨迹的起点: ({x_traj_initial[0,0]:.3f}, {x_traj_initial[0,1]:.3f}, {x_traj_initial[0,2]:.3f})")
    print(f"参数集 {param_set+1} 初始轨迹的终点: ({x_traj_initial[-1,0]:.3f}, {x_traj_initial[-1,1]:.3f}, {x_traj_initial[-1,2]:.3f})")
    
    # 运行iLQR迭代
    for iter in range(10):
        # 前向模拟当前轨迹
        x_traj = traj_sim(x0, u_traj)
        
        # 计算当前轨迹的总损失
        total_loss = np.sum([loss(t*dt, x_traj[t], u_traj[t]) for t in range(tsteps)])
        loss_list.append(total_loss)
        
        # 获取下降方向
        v_traj = ilqr_iter(x0, u_traj)
        
        # Armijo线搜索参数
        gamma = 1.0  # 初始步长
        alpha = 1e-04
        beta = 0.5
        
        # 计算初始步长下的损失
        x_traj_new = traj_sim(x0, u_traj + gamma * v_traj)
        new_loss = np.sum([loss(t*dt, x_traj_new[t], u_traj[t] + gamma * v_traj[t]) for t in range(tsteps)])
        
        # 执行Armijo线搜索
        while new_loss > total_loss - alpha * gamma * np.sum(v_traj * v_traj):
            gamma = beta * gamma
            x_traj_new = traj_sim(x0, u_traj + gamma * v_traj)
            new_loss = np.sum([loss(t*dt, x_traj_new[t], u_traj[t] + gamma * v_traj[t]) for t in range(tsteps)])
            
            # 避免步长过小导致的无限循环
            if gamma < 1e-6:
                break
        
        # 更新下一次迭代的控制
        u_traj += gamma * v_traj
        
        # 如果损失不再显著减少，停止迭代
        if iter > 0 and abs(loss_list[-1] - loss_list[-2]) < 1e-2:
            break
    
    # 创建此参数集的图
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. 状态轨迹图 - 使用增强的可视化
    desired_traj = np.array([[2.0*t*dt / np.pi, 0.0, np.pi/2.0] for t in range(tsteps)])
    
    # 绘制初始轨迹 - 使用更粗的线和更明显的样式
    axs[0].plot(x_traj_initial[:,0], x_traj_initial[:,1], 
               linestyle='--', linewidth=2.5, color='blue', 
               label="Initial Trajectory")
    
    # 绘制其他轨迹
    axs[0].plot(desired_traj[:,0], desired_traj[:,1], 
               linestyle='-', linewidth=1.5, color='r', 
               label="Desired Trajectory")
    axs[0].plot(x_traj[:,0], x_traj[:,1], 
               linestyle='-', linewidth=1.5, color='g', 
               label="Converged Trajectory")
    
    # 添加网格使轨迹更容易看到
    axs[0].grid(True, linestyle=':', alpha=0.7)
    
    axs[0].set_title('State Trajectory')
    axs[0].legend(loc='upper right')
    axs[0].set_xlim(0, 4)
    axs[0].set_ylim(-2, 2)
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    
    # 2. 最优控制图
    axs[1].plot(time, u_traj[:,0], label='$u_1(t)$')
    axs[1].plot(time, u_traj[:,1], label='$u_2(t)$')
    axs[1].legend()
    axs[1].set_xlim(0, 6)
    axs[1].set_ylim(-3, 3)
    axs[1].set_title('Optimal Control')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Control')
    
    # 3. 目标值图
    iters = np.arange(len(loss_list))
    axs[2].plot(iters, loss_list)
    axs[2].set_title('Objective Value')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Objective')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.05)
    plt.savefig(f'problem3_{param_set+1}.png', dpi=300, bbox_inches='tight')
    plt.show()