"""
Exploratory Material Flow Modeling and Analysis (EMFMA)

Demonstration of different sampling methods and associated distortions.

Author: N. Weijenberg
Institutions: Leiden University, TU Delft, TNO
Created: July 7, 2025
"""

#%% Imports and Setup

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import beta, dirichlet
from scipy.optimize import minimize

# Define distinct colors for visualizations
colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2'   # Pink
]

#%% Parameter Setup

# Data used in construction of Figure G.1
samples = np.array([0.1, 0.3, 0.6])
DQIS = [
    [3, 3, 2, 3, 2],
    [4, 4, 4, 4, 4],
    [2, 4, 3, 1, 1]
]

#%% (Optional) Random Experiment

# def random_experiment(n):
#     """
#     Generates randomized Dirichlet samples and corresponding DQIS scores.
#     """
#     k = 5
#     samples = np.random.dirichlet(np.ones(n), size=1)[0]
#     DQIS = [np.random.randint(1, 5, k) for _ in range(n)]
#     return samples, DQIS

# samples, DQIS = random_experiment(3)

#%% --- Uncertainty Bound Computation Functions ---

def compute_uncertainty(DQIS):
    """
    Computes total uncertainty value from DQIS scores.
    
    Parameters:
    DQIS (list): Pedigree matrix scores in the order [Geo, Temp, Mat, Tech, Rel].

    Returns:
    float: Uncertainty value.
    """
    k = len(DQIS)
    CVs = np.zeros(k)
    CV_total = 0

    for i in range(k):
        CVs[i] = np.exp(2.21 * DQIS[i]) if i == k - 1 else np.exp(2.21 * (DQIS[i] - 1))
        CV_total += CVs[i]

    CV_total = 1.5 * math.sqrt(CV_total) / 100 * 2.45
    uncertainty = 2.45 * CV_total / 100 * 2.45
    return uncertainty

#%% --- Optimization Functions for Trimming ---

def min_trimming(lower, upper):
    """
    Finds minimum values for each TC while ensuring total sum equals 1.

    Returns:
    np.ndarray: Each row contains a valid set of TCs with one minimized.
    """
    n = len(lower)
    optimized = []

    for i in range(n):
        TCs = lower.copy()
        remaining = 1 - np.sum(TCs)

        while not np.isclose(remaining, 0):
            for j in range(n):
                if j != i and remaining > 0:
                    addition = min(upper[j] - TCs[j], remaining)
                    TCs[j] += addition
                    remaining -= addition

            if not np.isclose(remaining, 0):
                for j in range(n):
                    if j != i:
                        TCs[j] = lower[j]
                TCs[i] = min(upper[i], TCs[i] + 0.01)
                remaining = 1 - np.sum(TCs)

        if np.isclose(np.sum(TCs), 1):
            optimized.append(TCs)

    return np.array(optimized)

def max_trimming(lower, upper):
    """
    Finds maximum values for each TC while ensuring total sum equals 1.

    Returns:
    np.ndarray: Each row contains a valid set of TCs with one maximized.
    """
    n = len(lower)
    optimized = []

    for i in range(n):
        TCs = upper.copy()
        excess = np.sum(TCs) - 1

        while not np.isclose(excess, 0):
            for j in range(n):
                if j != i and excess > 0:
                    removal = min(TCs[j] - lower[j], excess)
                    TCs[j] -= removal
                    excess -= removal

            if not np.isclose(excess, 0):
                for j in range(n):
                    if j != i:
                        TCs[j] = upper[j]
                TCs[i] = max(lower[i], TCs[i] - 0.01)
                excess = np.sum(TCs) - 1

        if np.isclose(np.sum(TCs), 1):
            optimized.append(TCs)

    return np.array(optimized)

def compute_constraints(lower, upper):
    """
    Computes min/max feasible values for each TC under sum constraint.

    Returns:
    tuple: (min_values, max_values) for each TC.
    """
    n = len(lower)
    min_set = min_trimming(lower, upper)
    max_set = max_trimming(lower, upper)

    constraints_min = np.array([min_set[i][i] for i in range(n)])
    constraints_max = np.array([max_set[i][i] for i in range(n)])
    return constraints_min, constraints_max

#%% --- Sampling and Fitting Functions ---

def objective(alpha_params):
    """
    Objective function: minimize squared error between elicited histogram
    and predicted bin values from Beta distribution.
    
    Requires global variables: n, bin_edges, elicited_histograms.
    """
    alpha_params = np.maximum(alpha_params, 1e-6)  # Prevent non-positive values
    alpha_0 = np.sum(alpha_params)
    sse = 0.0

    for i in range(n):
        a_i = alpha_params[i]
        b_i = alpha_0 - a_i

        for j in range(10):
            p_model = beta.cdf(bin_edges[j+1], a_i, b_i) - beta.cdf(bin_edges[j], a_i, b_i)
            p_elicited = elicited_histograms[i][j]
            sse += (p_model - p_elicited) ** 2

    return sse

def gibbs_sampling(num_samples, lower, upper, burn_in=0, thinning=1):
    """
    Samples valid vectors x such that:
      - lower[i] <= x[i] <= upper[i]
      - sum(x) = 1

    Uses pairwise Gibbs sampling strategy.
    """
    n = len(lower)

    if np.sum(lower) > 1 or np.sum(upper) < 1:
        raise ValueError("Feasible region is empty!")

    x = np.array(lower, dtype=float)
    remainder = 1 - np.sum(x)
    gaps = np.array(upper) - x

    if np.sum(gaps) > 0:
        x += remainder * gaps / np.sum(gaps)

    assert np.all(x >= lower) and np.all(x <= upper) and np.isclose(np.sum(x), 1)

    samples = []
    iteration = 0

    while len(samples) < num_samples:
        iteration += 1

        for i in range(n):
            for j in range(i + 1, n):
                s = x[i] + x[j]
                new_lower = max(lower[i], s - upper[j])
                new_upper = min(upper[i], s - lower[j])

                new_x_i = np.random.uniform(new_lower, new_upper)
                x[i] = new_x_i
                x[j] = s - new_x_i

        if iteration >= burn_in and (iteration - burn_in) % thinning == 0:
            samples.append(np.copy(x))

    return np.array(samples)

#%% Block A: Kawecki2018 - Triangular sampling with normalization

# Compute uncertainty factors from Data Quality Indicator Scores
uncertainties = [compute_uncertainty(i) for i in DQIS]
n = len(samples)  # Number of transfer coefficients (TCs)

# Calculate lower and upper bounds, clipped to [0, 1]
lower = [max(0, samples[i] - uncertainties[i]) for i in range(n)]
upper = [min(1, samples[i] + uncertainties[i]) for i in range(n)]

# Generate triangular samples with rejection to ensure bounds [0, 1]
triangular_samples = []
size = 100000  # Number of samples per TC

for i in range(n):
    sample_count = 0
    triangular_sample = []

    while sample_count < size:
        # Generate samples using triangular distribution
        new_samples = np.random.triangular(lower[i], samples[i], upper[i], size=size - sample_count)

        # Keep only valid samples within [0, 1]
        valid_samples = new_samples[(new_samples >= 0) & (new_samples <= 1)]
        triangular_sample.extend(valid_samples)
        sample_count = len(triangular_sample)

    # Store samples (ensure consistent length)
    triangular_samples.append(np.array(triangular_sample[:size]))

# Normalize each sample vector to sum to 1
normalized_triangular_samples = []
for i in range(size):  # Loop over sample index
    total = sum(triangular_samples[j][i] for j in range(n))
    normalized_sample = [triangular_samples[j][i] / total for j in range(n)]
    normalized_triangular_samples.append(normalized_sample)

normalized_triangular_samples = np.array(normalized_triangular_samples)

#%% Block B: Dong2023 - Dirichlet fitting via least squares

ds = n  # Dimensions (equal to number of TCs)
nb = 10  # Number of histogram bins

elicited_histograms = []
for d in range(ds):
    # Build histograms from triangular samples
    counts, bin_edges = np.histogram(triangular_samples[d], bins=nb, range=(0, 1), density=True)
    elicited_histograms.append(counts)

# Prepare bin definitions
theta_bins = np.linspace(0, 1, nb + 1)
bin_centers = np.linspace(0.05, 0.95, nb)
bin_edges = np.linspace(0, 1, nb + 1)

# Fit Dirichlet parameters by minimizing squared error to expert histograms
initial_guess = samples # Initial guess for parameters are observed means
bounds = [(1e-6, None)] * ds # bounds prevent zero/negative alphas
result = minimize(objective, x0=initial_guess, bounds=bounds, method='COBYLA')
alpha_fitted = result.x
loss = result.fun

#%% Block C: EMFMA - 2-Block Gibbs sampling from uniform bounds

# Refine bounds for compatibility with Gibbs sampler
lower_new, upper_new = compute_constraints(lower, upper)

# Sample from conditional space using 2-blocked Gibbs algorithm
num_samples = 10000
samples = gibbs_sampling(num_samples, lower_new, upper_new)

#%% Plotting results from all 3 approaches (A, B, C)

fig, axs = plt.subplots(1, 3, figsize=(19, 6))
plt.rcParams.update({'font.size': 20})

# --- A. Kawecki2018 Triangular Sampling (normalized) ---
for i in range(n):
    axs[0].hist(triangular_samples[i], bins=30, color=colors[i], density=True, alpha=0.5,
                label=f'Initial distribution (TC {i+1})', histtype='stepfilled')
    axs[0].hist(normalized_triangular_samples[:, i], bins=30, color=colors[i], density=True,
                label=f'Sampled distribution (TC {i+1})', histtype='step', linewidth=2)

axs[0].set_xlabel("Transfer coefficient")
axs[0].set_ylabel("Probability Density")
axs[0].set_title("A. Normalization of pedigree-\nbased triangular distributions")
axs[0].set_xlim([0, 1])
axs[0].set_ylim([0, 30])
axs[0].grid(True, axis='x', color='lightgray', linestyle='-', linewidth=0.5)
axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# --- B. Dong2023 Least Squares Dirichlet Fit ---
# Draw samples from fitted Dirichlet
dirichlet_samples = dirichlet(alpha_fitted).rvs(100000)

for d in range(ds):
    beta_marginal_samples = dirichlet_samples[:, d]
    
    # Plot sampled beta marginals
    axs[1].hist(beta_marginal_samples, bins=30, density=True,
                color=colors[d], histtype='step', linewidth=2)

    # Normalize expert histogram manually and plot
    bin_width = theta_bins[1] - theta_bins[0]
    normalized_hist = elicited_histograms[d] / (np.sum(elicited_histograms[d]) * bin_width)
    axs[1].bar(theta_bins[:-1], normalized_hist, width=bin_width, alpha=0.5, color=colors[d])

axs[1].set_xlabel("Transfer coefficient")
axs[1].set_title("B. Least-squares Dirichlet solution\nfor expert elicited histograms")
axs[1].set_xlim([0, 1])
axs[1].set_ylim([0, 30])
axs[1].grid(True, axis='x', color='lightgray', linestyle='-', linewidth=0.5)
axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# --- C. EMFMA Gibbs Sampling from Bounds ---
for i in range(n):
    x = np.linspace(lower_new[i], upper_new[i], 100)
    pdf = np.full_like(x, 1 / (upper_new[i] - lower_new[i]))  # Uniform PDF
    axs[2].fill_between(x, pdf, color=colors[i], alpha=0.5)  # Plot uniform prior
    axs[2].hist(samples[:, i], bins=30, density=True,
                color=colors[i], histtype='step', linewidth=2)

axs[2].set_xlabel("Transfer coefficient")
axs[2].set_title("C. Two-blocked Gibbs sampling\nof expert elicited bounds")
axs[2].set_xlim([0, 1])
axs[2].set_ylim([0, 30])
axs[2].grid(True, axis='x', color='lightgray', linestyle='-', linewidth=0.5)
axs[2].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# Final plot formatting
fig.legend(loc='center', bbox_to_anchor=(0.5, -0.05), ncol=n)
plt.tight_layout()
plt.show()
    