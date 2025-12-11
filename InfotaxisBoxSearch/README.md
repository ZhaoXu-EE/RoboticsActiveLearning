InfotaxisBoxSearch – Bayesian Infotaxis Agent for BoxSearch-v0
==============================================================

## 1. Overview

This directory contains an autonomous “box finder” implemented as an **infotaxis** search agent for the `BoxSearch-v0` Gym environment.  

The agent maintains a Bayesian belief over which box in a grid is "interesting" (e.g., contains a target) and chooses actions that **maximize the expected information gain**, measured via the reduction in Shannon entropy of the belief. The implementation is intentionally compact and self‑contained to serve as both a working demo and a pedagogical reference.

The core logic is implemented in a single file:

- `final.py`: infotaxis agent, belief update, entropy computation, and search loop.

## 2. Problem Formulation

### 2.1 Environment

The agent interacts with a discrete Gym environment:

- **Environment name**: `BoxSearch-v0`
- **State** (conceptual):
  - Robot pose (discrete grid position and possibly orientation).
  - Index of the currently sensed box.
  - Binary measurement indicating whether a positive signal is observed.
- **Action space**:
  - A finite set of legal motions of the robot’s “arm” or end effector (e.g., moves between neighboring boxes), obtained via `env.unwrapped.legal_actions(robot_pose)`.

Internally, the environment provides:

- `env.unwrapped.n_boxes`: number of candidate boxes.
- `env.unwrapped.prob_positive(robot_pose, box_id)`: generative model for the probability of a positive measurement when sensing box `box_id` from pose `robot_pose`.
- `env.unwrapped.predict_pose(robot_pose, action)`: deterministic prediction of the next pose under a given action.

### 2.2 Bayesian Belief

Let \(N\) denote the number of boxes. The agent maintains a **categorical belief** over boxes
\[
  b(i) = \Pr(\text{box } i \text{ is interesting}), \quad i = 1, \dots, N,
\]
subject to \(\sum_i b(i) = 1\). In the implementation:

- `belief` is a 1‑D NumPy array of length `n_boxes`.
- The prior is initialized as a uniform distribution:
  \[
    b_0(i) = \frac{1}{N}, \quad \forall i.
  \]

### 2.3 Measurement Model

For any robot pose `x` and box index `i`, the environment exposes
\[
  p_1(x, i) := \Pr(z = 1 \mid x, i),
\]
where \(z \in \{0,1\}\) is a binary measurement (e.g. 1 = positive signal, 0 = negative). The helper `likelihood(reading_one, robot_pose, box_id)` is a thin wrapper around this generative model and returns
\[
  \Pr(z = 1 \mid x, i) \quad \text{or} \quad \Pr(z = 0 \mid x, i) = 1 - p_1(x, i).
  \]

### 2.4 Bayesian Update

Suppose at time step \(t\), the agent senses box \(j\) from pose \(x_t\) and observes \(z_t \in \{0,1\}\). The belief over *which box is interesting* is updated via Bayes’ rule:
\[
  b_{t+1}(i) \propto b_t(i) \, \Pr(z_t \mid x_t, i),
\]
with normalization \(\sum_i b_{t+1}(i) = 1\).  

In `final.py`, this update is implemented in a lightweight way, exploiting the fact that at each step only a **single** box is probed:

- The measurement \(z_t\) is associated with some `box_idx`.
- Only `belief[box_idx]` is scaled by either \(p_1\) or \(1 - p_1\).
- The entire belief is then renormalized.

This preserves computational efficiency even for moderately large grids (tens of boxes).

## 3. Infotaxis Policy

### 3.1 Entropy as an Uncertainty Measure

The agent quantifies uncertainty in the belief using the **Shannon entropy**
\[
  H(b) = - \sum_{i=1}^{N} b(i) \log b(i),
\]
implemented in `entropy(b)`. To avoid numerical issues, logarithms are only taken for entries significantly greater than zero.

### 3.2 Look‑ahead: Expected Posterior Entropy

For each legal action \(a\) in the current pose \(x\), the agent:

1. Uses `env.unwrapped.predict_pose(x, a)` to compute the hypothetical next pose \(x'\).
2. For each box \(i\), queries \(p_1(x', i) = \Pr(z=1 \mid x', i)\).
3. Forms two hypothetical posterior beliefs:
   - \(b^{(1)}\): posterior if the next observation is \(z = 1\).
   - \(b^{(0)}\): posterior if the next observation is \(z = 0\).
4. Computes corresponding entropies \(H_1 = H(b^{(1)})\) and \(H_0 = H(b^{(0)})\).
5. Computes the **predictive probability of a positive measurement**
   \[
     p_{\text{meas} = 1} = \sum_i b(i) \, p_1(x', i),
   \]
   and thus \(p_{\text{meas} = 0} = 1 - p_{\text{meas} = 1}\).
6. Evaluates the **expected posterior entropy**
   \[
     \mathbb{E}[H(b')] = p_{\text{meas} = 1} H_1 + (1 - p_{\text{meas} = 1}) H_0.
   \]

This logic is encapsulated in the function `expected_entropy_after(action, robot_pose)`.

### 3.3 Action Selection: Maximizing Information Gain

Let \(H_{\text{curr}}\) denote the entropy of the current belief. For each candidate action \(a\), the **expected information gain** is defined as
\[
  \text{Gain}(a) = H_{\text{curr}} - \mathbb{E}[H(b' \mid a)].
  \]

The infotaxis policy is then:

- Enumerate all `legal_actions` from the current pose.
- For each action, compute `Gain(a)`.
- Select the action with the **maximum expected gain**.

In code, this is implemented in the main loop by:

1. Constructing a list of gains.
2. Selecting the index that maximizes the gain via `np.argmax`.
3. Executing the corresponding action in the environment.

This greedy one‑step look‑ahead is a common and effective approximation in infotaxis, balancing performance with computational tractability.

## 4. Algorithm Structure

At a high level, `final.py` follows three phases:

1. **Initialization**
   - Create the environment with `gym.make(ENV_NAME)`.
   - Reset the environment and read initial observation.
   - Initialize a uniform belief over boxes.
   - Initialize trajectory and entropy history containers.

2. **Search Loop**
   - For up to `max_steps` iterations:
     - Record the current pose.
     - Enumerate all legal actions from this pose.
     - For each action, compute expected posterior entropy and corresponding information gain.
     - Choose the action with maximal gain.
     - Step the environment, receive observation and reward.
     - Perform a Bayesian belief update conditioned on the measurement.
     - Append the new entropy to `ent_hist`.
     - Check termination:
       - Either the maximum box probability exceeds a high threshold (e.g. `belief.max() > 0.98`), or
       - The environment signals `done`.

3. **Post‑processing and Visualization**
   - Plot entropy over time using Matplotlib.
   - The plot provides a quick diagnostic of how rapidly the agent’s uncertainty collapses.

## 5. Usage

### 5.1 Dependencies

The code assumes:

- Python 3.8+ (not strict; any reasonably recent Python 3 version should work).
- `gym` (or `gymnasium`, depending on the local environment).
- `numpy`
- `matplotlib`
- A registered `BoxSearch-v0` environment (e.g., from a companion `box_gym_project` or similar package).

Install typical dependencies via:

```bash
pip install numpy matplotlib gym
```

You must also ensure that `BoxSearch-v0` is properly registered with Gym (e.g. by installing or adding the relevant environment package to your `PYTHONPATH`).

### 5.2 Running the Agent

From within the `InfotaxisBoxSearch` directory, run:

```bash
python final.py
```

Expected behavior:

- The script will:
  - Instantiate the environment.
  - Run the infotaxis agent until convergence or `max_steps` is reached.
  - Print a message indicating at which step convergence was achieved (if any).
  - Display a plot of Shannon entropy versus time step.

If run in a headless environment, you may wish to:

- Switch Matplotlib to a non‑interactive backend, or
- Replace `plt.show()` with `plt.savefig("entropy_curve.png", dpi=300)`.

## 6. Design Choices and Limitations

### 6.1 Design Choices

- **Belief representation**: a simple 1‑D categorical distribution over boxes, which is memory‑efficient and easy to update.
- **Exact enumeration of actions**: all legal actions are evaluated at each step, which is tractable for modest action spaces.
- **One‑step look‑ahead**: the agent optimizes expected information gain at the next time step only, avoiding the complexity of multi‑step planning.
- **Environment‑provided generative model**: by delegating signal generation to `env.unwrapped.prob_positive`, the agent remains agnostic to the specific sensor physics.

### 6.2 Limitations

- **Scalability**: For very large numbers of boxes (`n_boxes >> 50`) or dense action sets, the exhaustive evaluation of all actions may become computationally expensive.
- **Myopic policy**: The one‑step greedy infotaxis rule may be sub‑optimal in environments where long‑term planning is critical, though it typically performs well in practice.
- **Environment dependence**: The implementation assumes access to environment internals (`unwrapped` attributes). Porting to a different environment may require adapting these interfaces.

## 7. Conceptual Background

This project builds on several key ideas in probabilistic robotics and control:

- **Bayesian filtering** and belief updates.
- **Information‑theoretic objective functions** (entropy, mutual information).
- **Search and exploration strategies** that go beyond naive random or greedy exploitation.

The code thus serves as a concrete, end‑to‑end example of applying Bayesian and information‑theoretic reasoning to autonomous search in a stochastic environment.

## 8. References

- Vergassola, M., Villermaux, E., & Shraiman, B. I. (2007). *"Infotaxis" as a strategy for searching without gradients.* Nature, 445(7126), 406–409.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
  (for background on reinforcement learning and decision making under uncertainty).


