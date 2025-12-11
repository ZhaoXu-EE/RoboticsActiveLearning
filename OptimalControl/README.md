OptimalControl – Optimal Control and Numerical Optimization
===========================================================

## 1. Overview

This directory contains material on **optimal control and numerical optimization**. The work has three main components:

1. A derivation of the **two‑point boundary value problem (TPBVP)** underlying an iterative linear quadratic regulator (iLQR) scheme.
2. Implementation of **gradient descent with Armijo line search** on a quadratic function.
3. Application of **iLQR** to a differential‑drive vehicle tracking problem, with a parametric study of cost weights and initial controls.

## 2. Files

- `Homework4-2.py`: gradient descent with Armijo line search for a 2‑D quadratic objective.
- `Homework4-3.py`: iLQR implementation for a differential‑drive robot.
- `Homework4.md` / `Homework4.pdf`: derivations, explanations, and figure captions.
- `problem2_trajectory.png`: optimization trajectory overlaid on contours of the objective function.
- `problem2_convergence.png`: evolution of the objective value versus iteration.
- `problem3_1.png`, `problem3_2.png`, `problem3_3.png`: iLQR results under three different parameter sets.

## 3. Problem 1 – TPBVP and Linearized Dynamics

The first part develops the linearized dynamics and adjoint equations that arise in iLQR for a general nonlinear system. Starting from:

- A cost functional with state and control penalties:

  $$
    J = \int_0^T \bigl( l(x(t), u(t)) \bigr)\,dt + \phi(x(T)),
  $$

- A nominal trajectory $(x^{[k]}(t), u^{[k]}(t))$,
- Linearizations of the dynamics and cost around the nominal trajectory,

the homework derives:

- The adjoint dynamics

  $$
    \dot{p}(t) = -A(t)^\top p(t) - a_z(t),
  $$

- The algebraic condition on the control

  $$
    p(t)^\top B(t) + b_v(t)^\top = 0,
  $$

  which yields the optimal control increment

  $$
    v(t) = -R_v^{-1} B(t)^\top p(t),
  $$

  where $R_v$ is the control cost weight.

By combining the state perturbation $z(t)$ and the adjoint $p(t)$, the system is expressed as a TPBVP of the form

$$
  \begin{bmatrix} \dot{z}(t) \\ \dot{p}(t) \end{bmatrix}
  =
  \begin{bmatrix}
    A(t) & -B(t)R_v^{-1} B(t)^\top \\
    -Q_z & -A(t)^\top
  \end{bmatrix}
  \begin{bmatrix} z(t) \\ p(t) \end{bmatrix}
  +
  \begin{bmatrix} 0 \\ -a_z(t) \end{bmatrix},
$$

with appropriate boundary conditions, where $Q_z$ is the state cost weight. The homework identifies the block matrices $M_{ij}$ and vectors $m_1, m_2$ that define this system.

## 4. Problem 2 – Gradient Descent with Armijo Line Search

### 4.1 Objective and Setup

The second part implements gradient descent with an **Armijo backtracking line search** on a quadratic objective

$$
  f(x) = 0.26(x_1^2 + x_2^2) - 0.46 x_1 x_2.
$$

The algorithm is initialized at

$$
  x^{(0)} = [-4,\ -2]^\top,
$$

with:

- Initial step size $\gamma_0 = 1.0$,
- Armijo parameters $\alpha = 10^{-4}$, $\beta = 0.5$,
- A fixed maximum number of iterations (e.g. 100).

### 4.2 Results

The file `problem2_trajectory.png` shows:

- The level sets of \(f(x)\),
- The optimization path (red curve),
- The starting point (blue) and final point (green).

The file `problem2_convergence.png` plots $f(x^{(k)})$ versus iteration $k$, revealing a monotonic decrease in the objective value consistent with the Armijo condition. The results confirm stable convergence of the method on this convex quadratic.

## 5. Problem 3 – iLQR for a Differential‑Drive Vehicle

### 5.1 Dynamics and Objective

The third problem applies iLQR to a **differential‑drive robot** with dynamics

$$
  \begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{\theta} \end{bmatrix}
  =
  \begin{bmatrix}
    \cos(\theta) u_1 \\
    \sin(\theta) u_1 \\
    u_2
  \end{bmatrix},
  \quad (x(0), y(0), \theta(0)) = (0, 0, \pi/2),
$$

over a time horizon $T = 2\pi$. The desired trajectory is

$$
  (x_d(t), y_d(t), \theta_d(t)) = \left(\tfrac{4t}{2\pi},\, 0,\, \tfrac{\pi}{2}\right),
$$
and the cost functional penalizes deviations from this trajectory as well as control effort, via quadratic weights on state and input and a terminal cost.

### 5.2 Parameter Studies

Three parameter sets are examined:

1. **Default parameters**:
   - Initial control trajectory $u(t) = [1.0,\ -0.5]$,
   - State weight $Q_x = \mathrm{diag}(95.0, 10.0, 2.0)$,
   - Control weight $R_u = \mathrm{diag}(4.0, 2.0)$,
   - Terminal weight $P_1 = \mathrm{diag}(20.0, 20.0, 5.0)$.
2. **Different initial control**:
   - Same weights as above,
   - Initial control trajectory $u(t) = [0.5,\ 0.2]$.
3. **Different cost weights**:
   - $Q_x = \mathrm{diag}(50.0, 20.0, 5.0)$,
   - $R_u = \mathrm{diag}(2.0, 1.0)$,
   - $P_1 = \mathrm{diag}(30.0, 30.0, 10.0)$,
   - Initial control trajectory $u(t) = [0.8,\ 0.0]$.

The figures `problem3_1.png`, `problem3_2.png`, and `problem3_3.png` display, for each parameter set:

- The initial and optimized trajectories in the plane,
- The optimal controls $u_1(t)$ and $u_2(t)$,
- The evolution of the objective value over iLQR iterations.

### 5.3 Observations

Across all parameter sets, iLQR converges successfully, but the resulting trajectories and controls differ:

- Different initial controls yield distinct transient behavior and convergence paths.
- Modified cost weights shift the trade‑off between state tracking fidelity and control effort, leading to more oscillatory or more conservative control profiles.

These experiments highlight **sensitivity to initialization and weighting** in optimal control design, emphasizing the need for careful parameter tuning in practice.

## 6. Relation to Course Themes

This assignment connects:

- Analytical derivation of optimality conditions (TPBVPs and adjoint equations),
- Numerical optimization methods (gradient descent with line search),
- And advanced optimal control algorithms (iLQR) applied to nonlinear robotic systems.

Together, these components provide a concrete bridge between control theory and implementable algorithms for trajectory optimization.


