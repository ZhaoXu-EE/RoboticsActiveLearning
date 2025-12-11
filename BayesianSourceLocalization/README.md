BayesianSourceLocalization – Bayesian Source Localization Module
================================================================

## 1. Overview

This directory contains material on **Bayesian source localization** in a two‑dimensional domain. The exercises investigate how noisy binary measurements from spatially distributed sensors can be used to infer the location of an unknown source, both in batch form and via sequential Bayesian updating.

The work combines:

- A parametric measurement model \(p(z = 1 \mid x, s)\) that encodes how likely a sensor at position \(x\) observes a positive signal given a source at \(s\).
- Monte Carlo simulation of sensor placements and measurements.
- Grid‑based approximation of likelihoods and posterior beliefs over the source location.

## 2. Files

- `Homework1.py`: Python implementation of all five problems, including visualization scripts.
- `Homework1.md` / `Homework1.pdf`: written report and figure descriptions.
- `problem1_visualization.png`: measurement visualization for randomly placed sensors.
- `problem2_likelihood.png`: estimated likelihood map over candidate source locations.
- `problem3_trial_1.png`, `problem3_trial_2.png`, `problem3_trial_3.png`: fixed‑sensor trials with different sensor locations.
- `problem4_bayesian_update.png`: sequential Bayesian updates at a fixed sensor location.
- `problem5_moving_sensor.png`: sequential Bayesian updates with a moving sensor.

## 3. Problem Descriptions

### 3.1 Problem 1 – Measurement Visualization

The first task visualizes 100 random sensor locations in the unit square \([0, 1] \times [0, 1]\) relative to a fixed source location. Each sensor records a binary measurement \(z \in \{0,1\}\) drawn from the measurement model
\[
  p(z = 1 \mid x, s) = \exp\bigl(-100(\|x - s\| - 0.2)^2\bigr),
\]
which produces a ring‑shaped region of high positive‑signal probability around the source. The resulting plot overlays:

- The true source (blue “X”),
- Positive measurements (green dots),
- Negative measurements (red dots),
- A grayscale background indicating \(p(z=1\mid x,s)\).

This provides intuition about the spatial structure of the measurement model.

### 3.2 Problem 2 – Likelihood Estimation

Given the 100 measurements from Problem 1, Problem 2 constructs a **discrete likelihood map** over a grid of candidate source locations \(s\) in \([0,1]^2\). For each grid point, the likelihood is
\[
  L(s) = \prod_{i=1}^{N} \Pr(z_i \mid x_i, s),
\]
where \(N = 100\) is the number of measurements, \(x_i\) is the sensor position, and \(z_i\) is the observed binary signal. The product is computed numerically and normalized for visualization, yielding a heat map that concentrates around the true source location.

### 3.3 Problem 3 – Fixed Location Trials

In Problem 3, the sensor is no longer randomly placed at each measurement. Instead, three **fixed sensor locations** are selected, and for each:

- 100 measurements are drawn using the same measurement model.
- A likelihood map over candidate sources \(s\) is computed as in Problem 2.

The corresponding figures show how the informativeness of measurements depends on the sensor’s spatial placement and illustrate the ability of even a single static sensor to localize the source when enough measurements are accumulated.

### 3.4 Problem 4 – Sequential Bayesian Update at a Fixed Location

Problem 4 reformulates source localization as **sequential Bayesian inference**. For a fixed sensor location \(x\):

1. An initial **uniform prior** over source positions on a grid is specified.
2. A sequence of 10 measurements \(\{z_1, \dots, z_{10}\}\) is collected.
3. After each measurement, the belief \(b_t(s) = \Pr(s \mid z_{1:t})\) is updated via
   \[
     b_{t}(s) \propto b_{t-1}(s)\, \Pr(z_t \mid x, s),
   \]
   followed by normalization.

The resulting 2×5 panel figure shows how the belief becomes progressively more concentrated around the true source as more independent evidence is accumulated.

### 3.5 Problem 5 – Sequential Bayesian Update with a Moving Sensor

Finally, Problem 5 generalizes the sequential update to the case of a **moving sensor**, which visits a different random location at each time step:

- At step \(t\), the sensor is at \(x_t\) and observes \(z_t\).
- The belief update becomes
  \[
    b_t(s) \propto b_{t-1}(s)\, \Pr(z_t \mid x_t, s).
  \]

The associated 2×5 grid plot illustrates how a mobile sensor can actively explore the space while still driving the belief toward the true source location. Compared with the fixed‑sensor setting, mobility changes the spatial pattern of information acquisition but retains the same Bayesian update structure.

## 4. Relation to Course Themes

This assignment introduces several foundational ideas in probabilistic robotics:

- Construction and visualization of **measurement models**.
- Batch likelihood evaluation from multiple independent measurements.
- **Grid‑based Bayesian filtering** for static parameters (here, the source position).
- The role of sensor placement in the informativeness of observations.

These concepts underpin later assignments on information‑driven search and particle filtering.


