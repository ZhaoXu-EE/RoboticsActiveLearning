ParticleFilteringSampling – Sampling and Particle Filtering
===========================================================

## 1. Overview

This directory contains material on **Monte Carlo sampling and particle filtering**. The work has two main components:

1. **Rejection sampling** from a probability distribution defined implicitly by a grayscale image.
2. A **particle filter** for robot pose estimation, visualized via a one‑dimensional projection.

The work highlights how sampling‑based methods can represent complex, non‑Gaussian distributions and track uncertain dynamical systems.

## 2. Files

- `Homework3-1.py`: rejection sampling from an image‑based probability density.
- `Homework3-2.py`: particle filter implementation and visualization.
- `Homework3.md` / `Homework3.pdf`: written report and figure captions.
- `lincoln.jpg`: grayscale image used to construct the target distribution.
- `rejection_sampling.png`: visualization of samples drawn via rejection sampling.
- `pf_1d_projection.png`: comparison of prediction‑only and measurement‑updated particle filters.

## 3. Problem 1 – Rejection Sampling from an Image‑Based PDF

### 3.1 Target Distribution

An input grayscale image (`lincoln.jpg`) is treated as a **discrete probability map** over the unit square. Let $I(i,j)$ denote the pixel intensity at row $i$, column $j$. After inverting intensities so that darker pixels correspond to **higher probability**, a discrete mass function is obtained and normalized:

$$
  p_{ij} = \frac{\tilde{I}(i,j)}{\sum_{k,\ell} \tilde{I}(k,\ell)}, \qquad
  \tilde{I}(i,j) = 255 - I(i,j).
$$

Conceptually, this defines a continuous density $f(x,y)$ over $(x,y) \in [0,1]^2$ by mapping each coordinate to the corresponding pixel.

### 3.2 Rejection Sampling with Different Proposals

Two rejection samplers are implemented:

1. **Uniform proposal** over the unit square, where candidate samples $(x,y)$ are drawn uniformly and accepted with probability proportional to $f(x,y)$.
2. **Truncated Gaussian proposal**, where candidates are drawn from a Gaussian centered in the image (e.g. at $(0.5, 0.5)$) and rejected if they fall outside $[0,1]^2$.

For each proposal:

- A large number of candidate samples are generated.
- The target density $f(x,y)$ is evaluated and compared to a known upper bound $f_{\max}$.
- Candidates are accepted according to the rejection sampling criterion $u < f(x,y)/f_{\max}$, with $u \sim \mathcal{U}[0,1]$.

The final figure `rejection_sampling.png` shows:

- The underlying image‑based density (darker = higher probability).
- Scatter plots of 5000 accepted samples from each proposal.

Both samplers recover the facial structure present in the image, while the Gaussian proposal concentrates more strongly in high‑probability regions, yielding higher efficiency.

## 4. Problem 2 – Particle Filter: 1D Projection

### 4.1 State and Observation Models

The second problem implements a **particle filter** for robot pose estimation in a planar environment. The full state may include position and orientation, but the results are visualized via a one‑dimensional projection for clarity.

The particle filter is defined by:

- A **process model** $x_{t} = f(x_{t-1}, u_t, w_t)$, capturing robot motion with control input $u_t$ and process noise $w_t$.
- A **measurement model** $z_t = h(x_t, v_t)$, relating state to observations with measurement noise $v_t$.

Particles represent hypotheses about the robot state, and are updated in three stages at each time step:

1. **Prediction**: propagate each particle via the process model.
2. **Update**: weight particles by the likelihood of the observed measurement.
3. **Resampling**: draw a new particle set according to the normalized weights.

### 4.2 Visualization and Results

The figure `pf_1d_projection.png` compares two scenarios at selected time steps:

- **Left panel – prediction only**:
  - Only the process model is applied (no measurement updates).
  - The particle cloud gradually spreads and drifts due to accumulated process noise.
- **Right panel – prediction + measurement updates**:
  - Particles are regularly reweighted and resampled based on incoming measurements.
  - The cloud remains tightly concentrated around the true trajectory.

Orange circles indicate ground‑truth poses (with arrowheads indicating the heading), while the particle sets show uncertainty. The measurement‑updated filter exhibits markedly improved accuracy and stability, illustrating the importance of incorporating sensor information.

## 5. Relation to Course Themes

This assignment emphasizes:

- The use of **sampling** to represent and draw from complex distributions that are difficult to handle analytically.
- The structure and behavior of **particle filters** as recursive Bayesian estimators for nonlinear, non‑Gaussian systems.
- The visual interpretation of both static distributions (from images) and dynamic belief evolution (in state‑space models).

These tools form the foundation for more advanced stochastic state‑estimation techniques in robotics and control.


