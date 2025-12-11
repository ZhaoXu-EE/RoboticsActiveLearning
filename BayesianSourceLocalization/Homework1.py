import numpy as np
import matplotlib.pyplot as plt

# Problem 1: Measurement visualization
# Source location
source_location = np.array([0.3, 0.4])
num_samples = 100

# Function to compute p(z=Positive|x; s)
def p_positive(x, s):
    dist = np.linalg.norm(x - s, axis=-1)
    return np.exp(-100 * (dist - 0.2)**2)

# Sample 100 random locations in [0, 1] x [0, 1]
sample_locations = np.random.rand(num_samples, 2)

# Compute probabilities and sample measurements
probs = p_positive(sample_locations, source_location)
measurements = np.random.rand(num_samples) < probs  # True = Positive, False = Negative

# Visualization
fig, ax = plt.subplots(figsize=(6, 6))

# Background "ring" visualization
grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
grid_points = np.stack([grid_x, grid_y], axis=-1)
grid_probs = p_positive(grid_points, source_location)
ax.imshow(grid_probs, extent=(0,1,0,1), origin='lower', cmap='gray', alpha=0.5)

# Plot measurements
positive_points = sample_locations[measurements]
negative_points = sample_locations[~measurements]
ax.scatter(positive_points[:, 0], positive_points[:, 1], color='green', label='Positive Signal')
ax.scatter(negative_points[:, 0], negative_points[:, 1], color='red', label='Negative Signal')

# Plot source location as blue X
ax.plot(source_location[0], source_location[1], 'bx', markersize=10, markeredgewidth=2, label='Source')

# Add legend in white box at top right
legend_elements = [
    plt.Line2D([0], [0], color='b', marker='x', linestyle='None', markersize=10, label='Source'),
    plt.Line2D([0], [0], color='g', marker='o', linestyle='None', markersize=6, label='Positive Signal'),
    plt.Line2D([0], [0], color='r', marker='o', linestyle='None', markersize=6, label='Negative Signal')
]
legend = ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', edgecolor='black')
ax.set_title("Problem 1: Measurement Visualization")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
# plt.show()
# plt.savefig('problem1_visualization.png', dpi=300)











# Problem 2: Likelihood function estimation
grid_res = 100
grid_x, grid_y = np.meshgrid(np.linspace(0, 1, grid_res), np.linspace(0, 1, grid_res))
grid_s = np.stack([grid_x, grid_y], axis=-1)

# Compute likelihood L(s) for each grid point
likelihood = np.ones((grid_res, grid_res))
for i in range(num_samples):
    xi = sample_locations[i]
    zi = measurements[i]
    prob = p_positive(grid_s, xi)
    prob = prob if zi else (1 - prob)
    likelihood *= prob

# Normalize for visualization
likelihood /= np.max(likelihood)

# Visualization
fig, ax = plt.subplots(figsize=(6, 6))
likelihood_img = ax.imshow(likelihood, extent=(0,1,0,1), origin='lower', cmap='viridis', alpha=0.7)
fig.colorbar(likelihood_img, ax=ax, label='Likelihood')

# Plot measurements again
ax.scatter(positive_points[:, 0], positive_points[:, 1], color='green')
ax.scatter(negative_points[:, 0], negative_points[:, 1], color='red')

# Plot source location
ax.plot(source_location[0], source_location[1], 'bx', markersize=10, markeredgewidth=2)

# Add legend
legend = ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', edgecolor='black')
ax.set_title("Problem 2: Likelihood Estimation")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
# plt.show()
# plt.savefig('problem2_likelihood.png', dpi=300)















# Problem 3: Same sensor location, 3 trials with different fixed locations
fixed_locations = np.random.rand(3, 2)
grid_res = 100
trial_results = []

for trial in range(3):
    fixed_loc = fixed_locations[trial]
    # Generate 100 measurements at this fixed location
    fixed_probs = p_positive(np.array([fixed_loc]), source_location)[0]
    fixed_measurements = np.random.rand(100) < fixed_probs

    # Likelihood over grid
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, grid_res), np.linspace(0, 1, grid_res))
    grid_s = np.stack([grid_x, grid_y], axis=-1)
    likelihood = np.ones((grid_res, grid_res))

    for i in range(100):
        prob = p_positive(grid_s, fixed_loc)
        prob = prob if fixed_measurements[i] else (1 - prob)
        likelihood *= prob

    likelihood /= np.max(likelihood)
    trial_results.append((likelihood, fixed_loc, fixed_measurements))

# Plot all 3
figs = []
for i, (likelihood, fixed_loc, fixed_measurements) in enumerate(trial_results):
    fig, ax = plt.subplots(figsize=(6, 6))
    likelihood_img = ax.imshow(likelihood, extent=(0,1,0,1), origin='lower', cmap='viridis', alpha=0.7)
    fig.colorbar(likelihood_img, ax=ax, label='Likelihood')

    # Plot the fixed sensor location's measurements
    num_pos = np.sum(fixed_measurements)
    num_neg = 100 - num_pos
    ax.scatter([fixed_loc[0]] * num_pos, [fixed_loc[1]] * num_pos, color='green')
    ax.scatter([fixed_loc[0]] * num_neg, [fixed_loc[1]] * num_neg, color='red')

    # Plot true source
    ax.plot(source_location[0], source_location[1], 'bx', markersize=10, markeredgewidth=2)
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', edgecolor='black')
    ax.set_title(f"Problem 3: Trial {i+1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    # plt.show()
    # plt.savefig(f'problem3_trial_{i+1}.png', dpi=300)
    # Store the figure for later use
    figs.append(fig)
    







# Problem 4: Bayesian update at fixed location

# Initialize again
fixed_loc = np.random.rand(2)
grid_res = 100
x_vals = np.linspace(0, 1, grid_res)
y_vals = np.linspace(0, 1, grid_res)
grid_x, grid_y = np.meshgrid(x_vals, y_vals)
grid_s = np.stack([grid_x, grid_y], axis=-1)

# Initial uniform belief
belief = np.ones((grid_res, grid_res))
belief /= belief.sum()

# Store 10 measurement results and beliefs
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for t in range(10):
    # Simulate new measurement
    prob_pos = p_positive(np.array([fixed_loc]), source_location)[0]
    z = np.random.rand() < prob_pos

    # Bayesian update
    p_z_given_s = p_positive(grid_s, fixed_loc)
    p_z_given_s = p_z_given_s if z else (1 - p_z_given_s)
    belief *= p_z_given_s
    belief /= belief.sum()

    # Plot
    ax = axes[t]
    ax.imshow(belief, extent=(0,1,0,1), origin='lower', cmap='viridis', alpha=0.8)

    # Plot source and sensor location
    ax.plot(source_location[0], source_location[1], 'bx', markersize=10, markeredgewidth=2)
    ax.plot(fixed_loc[0], fixed_loc[1], 'go' if z else 'ro', markersize=6)

    ax.set_title(f"Step {t+1}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

# Overall formatting
plt.suptitle("Problem 4: Bayesian Update at Fixed Location", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
# plt.savefig('problem4_bayesian_update.png', dpi=300)






# Redo Problem 5 with consistent style and layout

# Initialize
grid_res = 100
x_vals = np.linspace(0, 1, grid_res)
y_vals = np.linspace(0, 1, grid_res)
grid_x, grid_y = np.meshgrid(x_vals, y_vals)
grid_s = np.stack([grid_x, grid_y], axis=-1)

# Uniform initial belief
belief = np.ones((grid_res, grid_res))
belief /= belief.sum()

# Random moving sensor locations
sensor_locations = np.random.rand(10, 2)

# Sequential updates
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for t in range(10):
    sensor_loc = sensor_locations[t]
    prob_pos = p_positive(np.array([sensor_loc]), source_location)[0]
    z = np.random.rand() < prob_pos

    # Bayesian update
    p_z_given_s = p_positive(grid_s, sensor_loc)
    p_z_given_s = p_z_given_s if z else (1 - p_z_given_s)
    belief *= p_z_given_s
    belief /= belief.sum()

    # Plot
    ax = axes[t]
    ax.imshow(belief, extent=(0,1,0,1), origin='lower', cmap='viridis', alpha=0.8)
    ax.plot(source_location[0], source_location[1], 'bx', markersize=10, markeredgewidth=2)
    ax.plot(sensor_loc[0], sensor_loc[1], 'go' if z else 'ro', markersize=6)

    ax.set_title(f"Step {t+1}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Problem 5: Bayesian Update with Moving Sensor", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
plt.savefig('problem5_moving_sensor.png', dpi=300)


