ErgodicControl – Ergodic Control for Robotic Exploration
========================================================

## 1. Overview

This directory contains material on **ergodic control**. The work studies how to design control inputs so that a robot’s time‑averaged visitation statistics match a prescribed spatial distribution, using variants of the iterative linear quadratic regulator (iLQR) for different dynamical systems:

1. A **first‑order** (single‑integrator) system,
2. A **second‑order** (double‑integrator) system,
3. A **differential‑drive** (non‑holonomic) robot.

Ergodic control provides a rigorous framework for coverage and exploration tasks in robotics.

## 2. Files

- `Homework5-1.py`: ergodic control for a first‑order (single‑integrator) system.
- `Homework5-2.py`: ergodic control for a second‑order (double‑integrator) system.
- `Homework5-3.py`: ergodic control for a differential‑drive robot.
- `Homework5.md` / `Homework5.pdf`: written report and figure commentary.
- `HW5_Problem1.png`: trajectory, control inputs, and objective value for the first‑order system.
- `HW5_Problem2.png`: analogous plots for the second‑order system.
- `HW5_Problem3.png`: analogous plots for the differential‑drive system.
- `problem1_fourier_target.npz`: stored Fourier coefficients of the target spatial distribution.
- `test.py`: auxiliary testing or debugging script.

## 3. Ergodic Control Framework

### 3.1 Target Distribution and Ergodicity

Let \(\phi(x)\) denote a **target spatial density** over a domain \(\mathcal{X} \subset \mathbb{R}^d\) that encodes where the robot should spend time (e.g. a Gaussian mixture). The goal of ergodic control is to design a control signal \(u(t)\) such that the robot trajectory \(x(t)\) satisfies, in a time‑averaged sense,
\[
  \frac{1}{T} \int_0^T f(x(t))\, dt
  \approx
  \int_{\mathcal{X}} f(x)\, \phi(x)\, dx
\]
for a suitable class of test functions \(f\). In practice, both the trajectory and the target distribution are projected into a truncated **Fourier basis**, and a quadratic cost penalizes discrepancies between their coefficients.

### 3.2 iLQR for Ergodic Objectives

For each dynamical system, an objective of the form
\[
  J = \sum_{k} \Lambda_k \bigl( c_k(x(\cdot)) - \Phi_k \bigr)^2
      + \int_0^T u(t)^\top R u(t)\, dt
\]
is minimized, where:

- \(\Phi_k\) are the Fourier coefficients of \(\phi(x)\) (stored in `problem1_fourier_target.npz`),
- \(c_k(x(\cdot))\) are the time‑averaged Fourier coefficients of the trajectory,
- \(\Lambda_k\) are mode‑dependent weights,
- \(R\) is a positive‑definite control cost matrix.

iLQR is used to iteratively:

1. Linearize the dynamics and quadratize the cost around a nominal trajectory.
2. Solve the resulting LQR subproblem backward in time.
3. Perform a forward rollout with updated controls.
4. Repeat until convergence of the objective.

## 4. Problem 1 – First‑Order System

The first problem considers a **single‑integrator** model
\[
  \dot{x}(t) = u(t),
\]
in a planar domain with a Gaussian‑mixture target density. The figure `HW5_Problem1.png` presents:

- **Left panel**: the optimized trajectory overlaid on the target density, showing repeated sweeps through high‑probability regions, consistent with spatial ergodicity.
- **Center panel**: control inputs \(u_x(t)\) and \(u_y(t)\), which remain smooth and low‑amplitude, indicating energy‑efficient motion.
- **Right panel**: objective value versus iLQR iteration, with a rapid initial decrease followed by a plateau near zero, demonstrating successful convergence.

## 5. Problem 2 – Second‑Order (Double‑Integrator) System

The second problem upgrades the dynamics to a **double integrator**, where acceleration is the control input. Qualitatively:

- The robot’s trajectory is initially straighter and exhibits more inertia, only later curving to better match the target density.
- Control inputs \(u_1(t)\) and \(u_2(t)\) show higher‑frequency components, as they must regulate both position and velocity.

The figure `HW5_Problem2.png` reports:

- **Left panel**: trajectory versus target density, revealing dynamics‑induced constraints on motion.
- **Center panel**: acceleration controls, reflecting more aggressive corrections.
- **Right panel**: objective value quickly collapses to a small value within the first iteration and then flattens, indicating that the ergodic objective is quickly satisfied under the chosen discretization and horizon.

## 6. Problem 3 – Differential‑Drive Robot

The third problem addresses a **non‑holonomic** differential‑drive robot. The state includes planar position and heading, while the control inputs are:

- Forward velocity \(v(t)\),
- Angular velocity \(\omega(t)\).

The figure `HW5_Problem3.png` shows:

- **Left panel**: a piecewise‑linear‑like closed loop that sweeps the main modes of the Gaussian mixture while respecting non‑holonomic constraints.
- **Center panel**: time histories of \(v(t)\) and \(\omega(t)\), with coupled bursts of speed and steering required to pivot and realign with ergodic targets.
- **Right panel**: the objective value over six iLQR iterations, decreasing monotonically and stabilizing at a low level, confirming convergence to an ergodic solution.

## 7. Relation to Course Themes

This assignment illustrates how:

- **Optimal control** (via iLQR) can be adapted to nonstandard objectives like spatial ergodicity.
- Different **dynamical models** (first‑order, second‑order, non‑holonomic) influence the achievable coverage patterns and control effort.
- Fourier‑based representations of distributions can be combined with trajectory optimization to design informative exploratory behaviors.

Together with earlier homeworks on Bayesian estimation and infotaxis, these results provide a unified view of how information, control, and dynamics interact in robotic exploration.


