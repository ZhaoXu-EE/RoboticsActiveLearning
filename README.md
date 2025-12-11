# ME455 – Robotic Sensing and Control Portfolio

## 1. Overview

This repository collects a sequence of projects and homework assignments for the course **ME455 – Robotic Sensing and Control** (MurpheyLab). Taken together, they form a compact portfolio covering:

- **Bayesian modeling and inference** for source localization,  
- **Information-driven exploration** and **infotaxis**,  
- **Sampling-based inference** (rejection sampling, particle filtering),  
- **Optimal control and trajectory optimization** (gradient methods, iLQR),  
- **Ergodic control** for coverage and exploration,  
- A final project implementing an **infotaxis search agent** in a custom Gym environment.

The six subdirectories each correspond to a focused module that builds on earlier concepts and tools.

## 2. Repository Structure

- `BayesianSourceLocalization/` – Homework 1: Bayesian source localization in 2D using batch likelihoods and sequential updates.  
- `InformationDrivenSearch/` – Homework 2: exploration strategies and infotaxis on a discrete grid.  
- `ParticleFilteringSampling/` – Homework 3: Monte Carlo sampling and particle filtering.  
- `OptimalControl/` – Homework 4: optimal control, gradient descent with line search, and iLQR for a differential-drive robot.  
- `ErgodicControl/` – Homework 5: ergodic control for first-order, second-order, and non-holonomic systems.  
- `InfotaxisBoxSearch/` – Final project: Bayesian infotaxis agent for the `BoxSearch-v0` Gym environment.

Each subdirectory contains:

- A **Python implementation** (`*.py`) of the main algorithms,  
- A **written report** (`Homework*.md`/`Homework*.pdf` or `README.md`) describing methods and results,  
- **Figures** (`*.png`, `*.jpg`) illustrating key behaviors and outcomes.

## 3. Thematic Progression

### 3.1 Probabilistic Sensing and Estimation

The first three modules emphasize probabilistic reasoning and sampling:

- **BayesianSourceLocalization** introduces a parametric measurement model and grid-based Bayesian inference to localize a static source from binary sensor readings, both in batch (likelihood maps) and sequential (Bayesian filtering) form.  
- **InformationDrivenSearch** builds on these ideas to design exploration policies that explicitly manage the exploration–exploitation trade-off, including a principled infotaxis controller that maximizes expected information gain.  
- **ParticleFilteringSampling** deepens the treatment of uncertainty by using sampling-based methods to approximate complex distributions and track robot poses under non-linear, noisy dynamics.

### 3.2 Optimal and Ergodic Control

The next two modules transition from estimation to control:

- **OptimalControl** develops the theory and practice of optimal control, from analytical derivation of two-point boundary value problems to numerical algorithms (gradient descent with Armijo line search) and iLQR applied to a differential-drive vehicle.  
- **ErgodicControl** adapts iLQR to ergodic objectives, using Fourier representations of target spatial distributions to design trajectories whose time-averaged statistics match prescribed densities under differing dynamics (single integrator, double integrator, differential drive).

### 3.3 Integrated Final Project

The **InfotaxisBoxSearch** final project integrates estimation and control in a realistic setting:

- The agent maintains a **Bayesian belief** over which box is “interesting”,  
- Uses an **entropy-based infotaxis policy** to select arm motions,  
- Interacts with a Gym environment (`BoxSearch-v0`) through a clean API,  
- And demonstrates convergence behavior via belief entropy plots.

This project serves as a capstone, unifying Bayesian modeling, information-theoretic objectives, and sequential decision making.

## 4. Dependencies and Execution

Most code in this repository is written in **Python** and relies on:

- `numpy` for numerical computation,  
- `matplotlib` for visualization,  
- `gym` (or a compatible RL environment library) for the final project and some search tasks,  
- Standard scientific Python tooling.

Because some environments (e.g. `BoxSearch-v0`) are course-specific, they must be installed or made available separately (e.g. via a companion `box_gym_project`).

To run an individual module:

1. Change into the corresponding subdirectory, e.g.:
   ```bash
   cd BayesianSourceLocalization
   ```
2. Consult the local `README.md` or homework write-up for problem-specific instructions.  
3. Execute the relevant script, e.g.:
   ```bash
   python Homework1.py
   ```

## 5. Educational Objectives

Collectively, these assignments are designed to:

- Provide hands-on experience with **Bayesian inference** and **Monte Carlo methods** in robotics,  
- Connect **information theory** (entropy, information gain) to exploration and search,  
- Demonstrate how **optimal control** and **trajectory optimization** are implemented in practice,  
- Illustrate the interplay between **dynamics**, **constraints**, and **coverage objectives** in ergodic control,  
- Culminate in an integrated project that reflects current research themes in robotic sensing and planning.

The repository may be used as a reference for students and practitioners interested in modern methods for robotic sensing, estimation, and control.


