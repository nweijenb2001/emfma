"""
Exploratory Material Flow Modeling and Analysis (EMFMA)
    
Demonstration of trimming.

Author: Weijenberg, N.
Institutions: Leiden University, TU Delft, TNO
Created: Jul7 7, 2025
"""

#%% Modules

# Import modules
import numpy as np # For data structures
import matplotlib.pyplot as plt # For plotting
import matplotlib.cm as cm # For colors

#%% Define the trimming algorithm

def min_trimming(lower, upper):
    """
    Optimizes the minimum values for each transfer coefficient (TC) within the
    uncertainty bounds while satisfying the sum constraint (sum of TCs equals 1).

    Parameters:
    lower (list or array): Lower uncertainty bounds for each TC.
    upper (list or array): Upper uncertainty bounds for each TC.

    Returns:
    np.array(optimized_combinations): Optimized combinations for each TC.
    """
    
    optimized_combinations = []
    n = len(lower)
    
    for i in range(n):  # Optimize each TC separately
        TCs = lower.copy()  # Start with minimum values for all TCs
        remaining_sum = 1 - np.sum(TCs)  # Calculate remaining sum to satisfy sum constraint
        
        while not remaining_sum == 0:  # Loop until the sum constraint is satisfied
            # Distribute the remaining sum to other TCs without exceeding their upper bounds
            for j in range(n):
                if j != i and remaining_sum > 0:  # Skip the TC being optimized
                    addable = min(upper[j] - TCs[j], remaining_sum)  # Determine how much can be added
                    TCs[j] += addable  # Add the value
                    remaining_sum -= addable  # Recalculate the remaining sum
                    
            # If we cannot satisfy the sum constraint, slightly increase the TC of interest and try again
            if not remaining_sum == 0:
                for j in range(n):
                    if j != i:
                        TCs[j] = lower[j]  # Reset other TCs to their minimum values
                TCs[i] = min(upper[i], TCs[i]+0.01)  # Slightly increase the TC being optimized
                remaining_sum = 1 - np.sum(TCs)  # Recalculate the remaining sum

        # Ensure that the sum of TCs is close to 1
        if np.isclose(np.sum(TCs), 1):
            optimized_combinations.append(TCs)  # Save the optimized combination
    
    return np.array(optimized_combinations)  # Return the optimized combinations as a numpy array

def max_trimming(lower, upper):
    """
    Optimizes the maximum values for each transfer component (TC) within the
    uncertainty bounds while satisfying the sum constraint (sum of TCs equals 1).

    Parameters:
    lower (list or array): Lower uncertainty bounds for each TC.
    upper (list or array): Upper uncertainty bounds for each TC.

    Returns:
    np.array(optimized_combinations): Optimized combinations for each TC.
    """
    
    optimized_combinations = []
    n = len(lower)
    
    for i in range(n):  # Optimize each TC separately
        TCs = upper.copy()  # Start with upper bounds for all TCs
        excess_sum = np.sum(TCs) - 1  # Calculate the excess sum that needs to be adjusted
        
        while not excess_sum == 0:  # Loop until the sum constraint is satisfied
            # Distribute the excess sum by reducing other TCs without going below their lower bounds
            for j in range(n):
                if j != i and excess_sum > 0:  # Skip the TC being optimized
                    removable = min(TCs[j] - lower[j], excess_sum)  # Determine how much can be reduced
                    TCs[j] -= removable  # Reduce the value
                    excess_sum -= removable  # Recalculate the excess sum
            
            # If we cannot satisfy the sum constraint, loosen the optimization goal slightly
            if not excess_sum == 0:
                for j in range(n):
                    if j != i:
                        TCs[j] = upper[j]  # Reset other TCs to their upper bounds
                TCs[i] = max(lower[i], TCs[i]-0.01)  # Slightly reduce the TC being optimized
                excess_sum = np.sum(TCs) - 1  # Recalculate the excess sumt

        # Ensure that the sum of TCs is close to 1
        if np.isclose(np.sum(TCs), 1):
            optimized_combinations.append(TCs)  # Save the optimized combination
    
    return np.array(optimized_combinations)  # Return the optimized combinations as a numpy array

def compute_constraints(lower, upper):
    """
    Computes the sum constraints based on optimized minimum and maximum values for each TC.

    Parameters:
    lower (list or array): Lower uncertainty bounds for each TC.
    upper (list or array): Upper uncertainty bounds for each TC.

    Returns:
    tuple: Two arrays containing the optimized minimum and maximum values for each TC.
    """
    
    # Optimize min and max values for each TC
    optimized_min = min_trimming(lower, upper)
    optimized_max = max_trimming(lower, upper)

    return optimized_min, optimized_max  # Return the computed constraints for min and max


#%% Randomized example

# Create random lower and upper bounds for TCs
n = 3 # Number of TCs
samples = np.random.dirichlet(np.ones(n), size=1)[0] # Reference values
lower = np.random.uniform(0, samples) # Lower bounds below reference
upper = np.random.uniform(samples, 1) # Upper bounds above reference

# Compute the minimum and maximum mass-balance constraints for the given bounds
constraints_min, constraints_max = compute_constraints(lower, upper)

# Plotting parameters
x_labels = np.arange(n) # Define x-axis labels
plt.rcParams.update({'font.size': 20}) # Set font size
colors = cmap = cm.get_cmap('tab10') # Get colormap
plt.xlabel("TC") # X-axis label
plt.ylabel("Value") # Y-axis label
plt.xticks(range(len(x_labels))) # Tick marks
plt.xticks(ticks=range(len(x_labels)), labels=x_labels) # Tick marks
plt.ylim([0, 1]) # Y-axis limits

# Plot original lower and upper bounds as gray rectangles
for i in range(n): # Iterate over TCs
    plt.fill_betweenx([lower[i], upper[i]], i - 0.1, i + 0.1, color='gray',
                      alpha=0.2, label="FR" if i == 0 else "")

# Plot optimized samples to demarcate the constraints, connect them with lines
for row in range(len(constraints_min)):
    
    color = color = cmap(row % 10) # Get a color
    
    for i in range(n - 1):
        
        # Plot min constraints
        plt.scatter([i, i + 1], [constraints_min[row, i], # Scattered points
                                 constraints_min[row, i + 1]], color=color, s=10)
        plt.plot([i, i + 1], [constraints_min[row, i], # Lines
                              constraints_min[row, i + 1]], color=color, alpha=0.3)

        # Plot max constraints
        plt.scatter([i, i + 1], [constraints_max[row, i], # Scattered points
                                 constraints_max[row, i + 1]], color=color, s=10)
        plt.plot([i, i + 1], [constraints_max[row, i], # Lines
                              constraints_max[row, i + 1]], color=color, alpha=0.3)

# Show the figure
plt.show()