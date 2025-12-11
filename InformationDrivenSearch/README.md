InformationDrivenSearch – Information-Driven Exploration
========================================================

## 1. Overview

This directory contains material on **information‑driven exploration and infotaxis**. The work analyzes and implements search strategies that explicitly trade off exploration and exploitation using Bayesian beliefs and information‑theoretic metrics.

The work comprises:

- An \(\varepsilon\)-greedy **exploration strategy** with decaying \(\varepsilon\) on a discrete grid.
- An **infotaxis** controller that selects actions to maximize expected information gain, demonstrated over multiple trials.

## 2. Files

- `Homework2-1.py`: implementation of the exploration strategy with a decaying \(\varepsilon\)-greedy policy.
- `Homework2-2.py`: implementation of the infotaxis controller and trial simulations.
- `Homework2.md` / `Homework2.pdf`: written report and figure descriptions.
- `exploration_results.png`: trajectory and belief evolution under the exploration strategy.
- `infotaxis_trial_1.png`, `infotaxis_trial_2.png`, `infotaxis_trial_3.png`: three infotaxis experiments with trajectory and belief visualizations.

## 3. Task 1 – Exploration Strategy with Decaying \(\varepsilon\)

### 3.1 Problem Setting

A robot moves on a discrete 2‑D grid, attempting to localize an unknown source that emits binary signals. At each time step, the robot:

1. Chooses an action from a finite set (e.g. up, down, left, right), subject to grid boundaries.
2. Receives a binary measurement \(z_t \in \{0,1\}\) whose probability depends on the robot’s position relative to the true source.
3. Updates its belief over candidate source locations.

### 3.2 \(\varepsilon\)-Greedy Policy

The policy for selecting actions is **\(\varepsilon\)-greedy** with a **decaying exploration rate**:

- With probability \(\varepsilon_t\), the agent **explores**, prioritizing previously unvisited neighboring cells to expand coverage of the grid.
- With probability \(1 - \varepsilon_t\), the agent **exploits**, moving toward regions with higher posterior belief.

The exploration rate follows a schedule
\[
  \varepsilon_t = \varepsilon_0 \cdot \gamma^t,
\]
with \(\varepsilon_0 \in (0,1]\) and decay factor \(\gamma \in (0,1)\), ensuring that the agent explores aggressively early on and gradually shifts to exploitation as the belief becomes more concentrated.

### 3.3 Results

The `exploration_results.png` figure shows:

- The robot’s path over the grid.
- The evolution of the belief over source locations.

The trajectory exhibits:

- Wide spatial coverage at early times (exploration),
- Smooth convergence toward the region of high belief,
- Avoidance of oscillatory behavior due to the decaying exploration rate.

Overall, the strategy reliably localizes the source while maintaining a clear exploration–exploitation balance.

## 4. Task 2 – Infotaxis Trials

### 4.1 Infotaxis Objective

In the infotaxis formulation, the robot’s goal is to **maximize expected information gain** about the source location at each step. Let \(b_t(i)\) denote the belief that grid cell \(i\) contains the source at time \(t\). The uncertainty is quantified via the Shannon entropy
\[
  H(b_t) = - \sum_i b_t(i) \log b_t(i).
\]

For each candidate action \(a\), the agent:

1. Predicts the next robot position \(x_{t+1}\) under \(a\).
2. Computes, for every candidate source location, the probability of obtaining a positive measurement.
3. Constructs hypothetical posterior beliefs conditioned on \(z_{t+1} = 1\) and \(z_{t+1} = 0\).
4. Evaluates the corresponding entropies \(H_1\) and \(H_0\).
5. Forms the **expected posterior entropy**
   \[
     \mathbb{E}[H(b_{t+1}) \mid a] = p(z_{t+1}=1 \mid a)\,H_1 + p(z_{t+1}=0 \mid a)\,H_0.
   \]

The infotaxis policy then chooses the action that **minimizes** the expected posterior entropy, equivalently **maximizing** the expected information gain
\[
  \text{Gain}(a) = H(b_t) - \mathbb{E}[H(b_{t+1}) \mid a].
\]

### 4.2 Simulation Trials

Three independent infotaxis trials are performed, differing in random seeds and initial conditions. The figures:

- `infotaxis_trial_1.png`
- `infotaxis_trial_2.png`
- `infotaxis_trial_3.png`

illustrate typical behavior:

- **Trial 1**: relatively symmetric exploration followed by rapid convergence to the source.
- **Trial 2**: pronounced horizontal exploration before the belief collapses in the upper‑right region.
- **Trial 3**: early exploration of left and bottom areas, then a directed move toward the central target.

In all cases, the **entropy decreases steadily** as the belief mass concentrates near the true source, confirming that the infotaxis policy is effective at guiding the search.

## 5. Comparison and Insights

The assignment contrasts:

- A heuristic yet effective **decaying \(\varepsilon\)-greedy** exploration policy, and
- A principled, information‑theoretic **infotaxis** controller.

Both strategies are capable of localizing the source, but infotaxis typically:

- Achieves faster convergence,
- Exhibits more structured trajectories,
- Provides a clearer connection to Bayesian decision‑making.

These experiments provide a bridge from simple exploration heuristics to information‑driven policies that will be further developed in the final project.


