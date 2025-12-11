import numpy as np

# 保存用于 Problem 1 的 Fourier basis 和目标密度，用于后续 iLQR 接入
def build_fourier_ergodic_terms(K=10, T=10.0, dt=0.1):
    # Fourier index
    k_list = []
    for i in range(K):
        for j in range(K):
            k_list.append(np.array([i, j]))
    k_list = np.array(k_list)
    num_k = k_list.shape[0]

    # Target density parameters
    weights = [0.5, 0.2, 0.3]
    mus = [np.array([0.35, 0.38]), np.array([0.68, 0.25]), np.array([0.56, 0.64])]
    covs = [
        np.array([[0.01, 0.004], [0.004, 0.01]]),
        np.array([[0.005, -0.003], [-0.003, 0.005]]),
        np.array([[0.008, 0.0], [0.0, 0.004]])
    ]

    def phi_k_scalar(k):
        value = 0.0
        k_scaled = (np.pi * k)  # domain L = [1,1]
        for w, mu, cov in zip(weights, mus, covs):
            exponent = -0.5 * k_scaled.T @ cov @ k_scaled
            cosine_term = np.cos(np.pi * k @ mu)
            value += w * np.exp(exponent) * cosine_term
        return value

    phi = np.array([phi_k_scalar(k) for k in k_list])
    return k_list, phi

k_list_for_ilqr, phi_target_for_ilqr = build_fourier_ergodic_terms(K=10)
np.savez("/home/xu/workspace/ME455/HW5/problem1_fourier_target.npz", k_list=k_list_for_ilqr, phi=phi_target_for_ilqr)
