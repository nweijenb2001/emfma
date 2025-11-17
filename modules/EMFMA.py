"""
Exploratory Material Flow Modeling and Analysis (EMFMA)

Supporting functions for EMFMA

Author: Weijenberg, N.
Institutions: Leiden University, TU Delft, TNO
Created: July 7, 2025
"""

#%% Modules and setup

import os  # For interaction with the operating system
import glob  # For finding all file pathnames matching a specified pattern
import re  # For pattern matching in strings with regular expressions

import numpy as np  # For numerical operations, arrays, and functions
import pandas as pd  # For data manipulation and analysis using DataFrames
import math # For determining whether numbers are close

import seaborn as sns  # For distributional plots
import matplotlib.pyplot as plt  # For elicitation & parallel coordinate plots
from matplotlib import cm  # For accessing color maps in matplotlib
import matplotlib.ticker as ticker  # For customizing axis ticks
import matplotlib.lines as mlines  # For creating custom lines in plots
import matplotlib.patches as patches # For background color

from plotly.subplots import make_subplots  # For creating subplots
import plotly.graph_objects as go  # For building interactive Sankey diagrams

from scipy.stats import spearmanr # For vulnerability analysis

from matplotlib.colors import Normalize, to_hex # For Sankey diagrams

#%% Elicitation

def load_elicitation(df, folder, prefix):
    '''
    Loads elicitation data from multiple Excel files into a dictionary.

    Parameters:
    - df: dict
        A dictionary that will be updated with data from the loaded Excel files.
    - folder: str
        Path to the folder containing the Excel files.
    - prefix: str
        Common prefix for the Excel filenames (e.g., "expert").

    Returns:
    - df
        Returns the `df` dictionary with parsed data from each Excel file.
    '''
    
    # Define row ranges for mfa and adm data
    mfa_rows = np.arange(0, 101, 1)
    adm_rows = np.arange(101, 148, 1)
    mfa_rows = np.append(mfa_rows, np.arange(148, 154, 1))
    adm_rows = np.append(adm_rows, np.arange(154, 176, 1))
    mfa_rows = np.append(mfa_rows, np.arange(176, 231, 1))
    adm_rows = np.append(adm_rows, np.arange(231, 246, 1))
    
    # Compile a pattern to extract expert elicitation files
    pattern = re.compile(rf'{prefix}_(\d+)\.xlsx')

    # Iterate over matching Excel files
    for filepath in sorted(glob.glob(os.path.join(folder, f'{prefix}_*.xlsx'))):
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match:
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore", UserWarning)
                i = int(match.group(1)) # Extract expert number

                # Load sheet names first
                sheet_names = pd.ExcelFile(filepath).sheet_names

                # Load individual sheets with proper header skipping
                sheets = {
                    sheet_names[1]: pd.read_excel(filepath, sheet_name=sheet_names[1], header=1),  # 1 row before header
                }

                df.setdefault(i, {}) # Initialize expert's entry in the df

                # --- Sheet 1: Transfer Coefficients (x_df) ---
                x_df = sheets[sheet_names[1]].copy()
                x_df = x_df[x_df.iloc[:, -2] == 1] # Filter for TRUE rows
                x_df = x_df[x_df['From'] != ""] # Exclude empty rows
                
                # Rename and select relevant columns
                colnames = x_df.columns.tolist()
                colnames[4] = 'Lower'
                colnames[5] = 'Upper'
                x_df.columns = colnames
                x_df = x_df[['From', 'To', 'Lower', 'Upper']]
                x_df = x_df.reset_index(drop=True)
                
                # Structure rows according to required format for MFA module
                mfa_x = x_df.loc[x_df.index.isin(mfa_rows)].copy()
                mfa_x['Country'] = 'Netherlands'
                mfa_x = mfa_x.set_index(['From', 'To', 'Country'])
                mfa_x = mfa_x.dropna()
                df[i]['mfa_x'] = mfa_x
                
                # Structure rows according to required format for ADM module
                adm_x = x_df.loc[x_df.index.isin(adm_rows)].copy()
                adm_x['Polymer'] = 'PET'
                adm_x = adm_x.set_index(['From', 'To', 'Polymer'])
                adm_x = adm_x.dropna()
                df[i]['adm_x'] = adm_x

    return df

def merge_and_trim_elicitation(elicitation):
    '''
    Merges and trims elicited transfer coefficient data across multiple experts.

    Parameters:
    - elicitation: dict
        A dictionary where each key is an expert ID, and each value is a sub-dictionary
        containing DataFrames for 'mfa_x', 'adm_x', 'mfa_p1', and 'mfa_p2'.
        Each DataFrame contains 'Lower' and 'Upper' bounds.

    Returns:
    - merged_elicitation: dict
        A dictionary with the same keys as the original sub-dictionaries (e.g., 'mfa_x', 'adm_x'),
        where each value is a DataFrame merged across experts, containing trimmed minimum Lower
        bounds and maximum Upper bounds.
    '''

    merged_elicitation = {}

    # Iterate over each category of data to merge
    for df in ['mfa_x', 'adm_x']:

        merged_bounds = pd.DataFrame()
        lower_columns = []  # Store names of temporary lower bound columns
        upper_columns = []  # Store names of temporary upper bound columns

        # Go through each expert's data
        for key, subdict in elicitation.items():
            if df in subdict:
                sub_df = subdict[df]

                suffix = f"_{key}"  # Suffix to distinguish columns by expert

                if merged_bounds.empty:
                    # Initialize the merged DataFrame from the first available one
                    merged_bounds = sub_df.copy()
                    if 'Lower' in sub_df.columns:
                        merged_bounds.rename(columns={'Lower': f'Lower{suffix}'}, inplace=True)
                        lower_columns.append(f'Lower{suffix}')
                    if 'Upper' in sub_df.columns:
                        merged_bounds.rename(columns={'Upper': f'Upper{suffix}'}, inplace=True)
                        upper_columns.append(f'Upper{suffix}')
                else:
                    # Add each expert's bounds as new columns
                    if 'Lower' in sub_df.columns:
                        col_name = f'Lower{suffix}'
                        merged_bounds[col_name] = sub_df['Lower']
                        lower_columns.append(col_name)
                    if 'Upper' in sub_df.columns:
                        col_name = f'Upper{suffix}'
                        merged_bounds[col_name] = sub_df['Upper']
                        upper_columns.append(col_name)
                    # Preserve context columns like 'Country' or 'Polymer' if they exist
                    if 'Country' in sub_df.columns:
                        merged_bounds['Country'] = sub_df['Country'].combine_first(merged_bounds.get('Country'))
                    if 'Polymer' in sub_df.columns:
                        merged_bounds['Polymer'] = sub_df['Polymer'].combine_first(merged_bounds.get('Polymer'))

        # Aggregate across experts by computing the union of lower and upper bounds
        if lower_columns:
            merged_bounds['Lower'] = merged_bounds[lower_columns].min(axis=1)
        if upper_columns:
            merged_bounds['Upper'] = merged_bounds[upper_columns].max(axis=1)

        merged_elicitation[df] = merged_bounds

    # Trim values within each process (i.e. 'From' group) to constrained ranges
    for category in ['adm_x', 'mfa_x']:
        df = merged_elicitation[category]
        grouped = df.groupby('From')

        for group_name, group in grouped:
            lower = group['Lower'].values
            upper = group['Upper'].values

            # Trim if group has more than one row and lower ≠ upper for any row
            if len(lower) > 1 and any(lower != upper):
                lower, upper = trimming(lower, upper)

            # Update trimmed values in the DataFrame
            df.loc[group.index, 'Lower'] = lower
            df.loc[group.index, 'Upper'] = upper

    return merged_elicitation

def plot_elicitation(merged_elicitation, folder):
    '''
    Plots horizontal bar charts to visualize uncertainty ranges for each parameter
    across different experts, as well as the final "trimmed union" bounds.

    Parameters:
    - merged_elicitation: dict
        Dictionary with keys like 'adm_x' and 'mfa_x', each containing a DataFrame
        that holds lower and upper uncertainty bounds from different experts and the final union.
    - folder: str
        Directory path where the generated plots will be saved.

    Function behavior:
    - Filters and splits the data into predefined thematic groups.
    - Skips plotting rows where all values are 0 or 1.
    - For each parameter:
        - If lower == upper → plots a marker.
        - If lower < upper → plots a horizontal bar.
    - Adds legend entries only if the expert's data is plotted.
    '''

    # Define distinct colors for up to 10 experts
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Row index ranges and names for each thematic block (for adm_x and mfa_x)
    split_ranges = {
        "adm_x": [(0, 13), (13, 29), (29, 47), (47, 62)],
        "mfa_x": [(0, 21), (21, 31), (31, 38), (38, 54), (54, 61),
                  (61, 74), (74, 78), (78, 88), (88, 104), (104, 122),
                  (122, 133)]
    }
    split_names = {
        "adm_x": [
            "Plastic degradation in the environment",
            "Biological uptake and dispersion of microplastics adm",
            "Dispersion and degradation of plastic litter",
            "Processes with two flows adm"
        ],
        "mfa_x": [
            "Plastic disposal behavior of consumers",
            "Plastic waste collection and treatment",
            "Plastic recycling",
            "Wastewater treatment",
            "Sewage sludge and compost application",
            "Cleaning and dispersion of plastic litter",
            "Biological uptake and dispersion of microplastics mfa",
            "Trivial/given data",
            "Landscape data",
            "Duplicate data",
            "Processes with two flows mfa"
        ]
    }

    # Loop over relevant datasets
    for df_name, df in merged_elicitation.items():
        if df_name not in ['mfa_x', 'adm_x']:
            break

        # Get all expert-specific and final Lower/Upper columns
        lower_cols = [col for col in df.columns if col.startswith('Lower_')]
        upper_cols = [col.replace('Lower', 'Upper') for col in lower_cols]
        lower_cols.append('Lower')  # Final trimmed bounds
        upper_cols.append('Upper')

        # Get thematic group names and ranges
        ranges = split_ranges.get(df_name, [(0, None)])
        names = split_names.get(df_name, [f"Rows_{start}_{end}" for start, end in ranges])

        # Iterate over each thematic block
        for (start, end), name in zip(ranges, names):
            df_range = df.iloc[start:end] if end is not None else df.iloc[start:]
            keep_rows = []

            # Filter out rows where all values are 0.0 or 1.0 (not informative)
            for idx, row in df_range.iterrows():
                values = row[lower_cols + upper_cols].dropna().values
                if not (np.all(values == 0.0) or np.all(values == 1.0)):
                    keep_rows.append(idx)

            if not keep_rows:
                continue  # Skip plotting if no informative rows

            df_chunk = df.loc[keep_rows]
            categories = df_chunk.index.tolist()
            n_categories = len(categories)
            n_series = len(lower_cols)

            # Define layout parameters
            spacing = 1
            y_base = np.arange(n_categories) * spacing
            bar_width = 0.8 / n_series
            fig_height = max(4, n_categories * 0.4)
            fig_width = max(10, n_categories * 0.6)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            legend_added = {}  # Keep track of which labels are already shown in legend

            # Plot each expert (or the final range)
            pairs = list(zip(lower_cols, upper_cols))[:-1]  # Remove'Trimmed Union' pair with [:-1]
            for idx, (lc, uc) in enumerate(pairs):
                lower = df_chunk[lc].values
                upper = df_chunk[uc].values
                offset = (idx - (n_series - 1) / 2) * bar_width
                bar_pos = y_base + offset

                # Get expert ID or label
                person = lc.split('_')[1] if '_' in lc else 'Trimmed Union'
                label = f'Person {person}' if person != 'Trimmed Union' else 'Trimmed Union'

                color = colors[idx % len(colors)]
                has_bar = False
                has_marker = False

                # Draw bars or markers
                for j, (pos, low, up) in enumerate(zip(bar_pos, lower, upper)):
                    if not np.isnan(low) and not np.isnan(up):
                        if low == up:
                            ax.plot(low, pos, 'x', color=color, label="_nolegend_", markersize=6)
                            has_marker = True
                        else:
                            ax.barh(pos, up - low, left=low, height=bar_width,
                                    label=label if label not in legend_added else "", color=color)
                            legend_added[label] = True
                            has_bar = True

                # Add legend entry if no bars but markers exist
                if not has_bar and has_marker and label not in legend_added:
                    ax.plot([], [], 'x', color=color, label=label)
                    legend_added[label] = True
                elif not has_bar and not has_marker and label not in legend_added:
                    ax.plot([], [], 'x', color=color, label=label)
                    legend_added[label] = True

            # Final layout settings
            ax.set_yticks(y_base)
            ax.set_yticklabels([f"{c[0]} --> {c[1]}" for c in categories])
            ax.set_xlabel('Value Range')
            ax.set_title(f'Uncertainty Ranges for "{name}"')
            ax.legend()
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
            plt.xlim([0, 1])
            plt.tight_layout()

            # Save figure
            os.makedirs(folder, exist_ok=True)
            safe_name = name.lower().replace(" ", "_").replace("/", "_")
            plt.savefig(f'{folder}/elicitation_{safe_name}.png')
            plt.close()
            
def store_elicitation(merged_elicitation, folder):
    """
    Stores elicited data for each scenario (BAU and policies) into CSV files.

    Parameters:
    - merged_elicitation: dict of DataFrames)
        Contains different scenario data.
    - folder: str
        Path to the folder where CSVs should be stored.
    """

    for df_name, df in merged_elicitation.items():
            # Remove non_aggregated bounds of individual experts
            df = df.loc[:, ~df.columns.str.startswith('Lower_') & ~df.columns.str.startswith('Upper_')]
            df = df.reset_index()
            df.to_csv(os.path.join(folder, f'{df_name}.csv'), index=True)

#%% Loading data

def merge_uncertainties_and_levers(bounds, p):
    """
    Merges base uncertainty bounds with policy modifications for each module and policy.

    Parameters:
    - bounds : dict
        Nested dictionary of DataFrames structured as {policy_index: {module_name: DataFrame}}.
        Each DataFrame contains uncertainty or lever bounds for a given module.
    - p : int
        Total number of policy scenarios (including BAU as index 0).

    Notes:
    - Lever bounds overwrite baseline bounds where present.
    """
    
    for module in ['MFA', 'ADM']:
        
        # Retrieve baseline uncertainty bounds (policy 0 = BAU)
        bounds_x = bounds[0].get(module)
        if bounds_x is None:
            continue  # Skip if no base uncertainty is defined for this module
        
        for policy in range(p):
            if policy == 0:
                continue  # Skip base policy

            bounds_p = bounds[policy].get(module)
            if bounds_p is None:
                continue  # Skip if no lever bounds are defined for this policy

            # Determine merging keys depending on the DataFrame columns
            merge_keys = ['From', 'To']
            if 'Country' in bounds_x.columns:
                merge_keys.append('Country')
            elif 'Polymer' in bounds_x.columns:
                merge_keys.append('Polymer')

            # Merge base and lever bounds
            merged_bounds = bounds_x.merge(
                bounds_p,
                on=merge_keys,
                how='left',
                suffixes=('_x', '_p')
            )

            # Replace base bounds with policy lever values where available
            for col in ['Lower', 'Upper']:
                merged_bounds[col] = merged_bounds[col + '_p'].fillna(merged_bounds[col + '_x'])
                merged_bounds[col] = merged_bounds[col].astype(float)
                merged_bounds = merged_bounds.drop(columns=[col + '_p', col + '_x'])

            # Update the bounds dictionary with the merged DataFrame
            bounds[policy][module] = merged_bounds

#%% Trimming

def min_trimming(lower, upper):
    '''
    Determines the minimum value for each TC within its uncertainty bounds
    that still allows the set of TCs to add up to 1, given the uncertainty
    bounds of the other TCs.

    Parameters:
    lower (list or array): Lower uncertainty bounds for each TC.
    upper (list or array): Upper uncertainty bounds for each TC.

    Returns:
    constraints_min: Minimized values of TCs
    '''
    
    optimized_combinations = []
    n = len(lower)
    
    for i in range(n):  # Optimize each TC separately

        TCs = lower.copy()  # Start with minimum values for all TCs
        remaining_sum = 1 - np.sum(TCs)  # Calculate remaining sum to satisfy sum constraint

        while not math.isclose(remaining_sum, 0, abs_tol=1e-9):  # Loop until the sum constraint is satisfied
            
            # Distribute the remaining sum to other TCs without exceeding their upper bounds
            for j in range(n):
                
                if j != i and remaining_sum > 0:  # Skip the TC being optimized
                
                    addable = min(upper[j] - TCs[j], remaining_sum)  # Determine how much can be added
                    TCs[j] += addable  # Add the value
                    remaining_sum -= addable  # Recalculate the remaining sum
                    
            # If we cannot satisfy the sum constraint, slightly increase the TC of interest before trying again
            if not math.isclose(remaining_sum, 0, abs_tol=1e-9):
                
                for j in range(n):
                    
                    if j != i:
                        
                        TCs[j] = lower[j]  # Reset other TCs to their minimum values
                        
                TCs[i] = min(upper[i], TCs[i]+0.0001)  # Slightly increase the TC being optimized
                remaining_sum = 1 - np.sum(TCs)  # Recalculate the remaining sum
                
        # Ensure that the sum of TCs is close to 1
        if np.isclose(np.sum(TCs), 1):
            
            optimized_combinations.append(TCs)  # Save the optimized combination
    
    # Extract the individual optimized TCs (diagonal elements)
    optimized_min = np.array(optimized_combinations)
    constraints_min = np.array([optimized_min[i][i] for i in range(n)])
    
    return constraints_min  # Return the optimized values

def max_trimming(lower, upper):
    '''
    Determines the maximum value for each TC within its uncertainty bounds
    that still allows the set of TCs to add up to 1, given the uncertainty
    bounds of the other TCs.

    Parameters:
    lower (list or array): Lower uncertainty bounds for each TC.
    upper (list or array): Upper uncertainty bounds for each TC.

    Returns:
    constraints_max: Maximized values of TCs
    '''
    
    optimized_combinations = []
    n = len(lower)
    
    for i in range(n):  # Optimize each TC separately
    
        TCs = upper.copy()  # Start with upper bounds for all TCs
        excess_sum = np.sum(TCs) - 1  # Calculate the excess sum that needs to be adjusted
        
        while not math.isclose(excess_sum, 0, abs_tol=1e-9):  # Loop until the sum constraint is satisfied
        
            # Distribute the excess sum by reducing other TCs without going below their lower bounds
            for j in range(n):
                
                if j != i and excess_sum > 0:  # Skip the TC being optimized
                
                    removable = min(TCs[j] - lower[j], excess_sum)  # Determine how much can be reduced
                    TCs[j] -= removable  # Reduce the value
                    excess_sum -= removable  # Recalculate the excess sum
            
            # If we cannot satisfy the sum constraint, loosen the optimization goal slightly
            if not math.isclose(excess_sum, 0, abs_tol=1e-9):
                
                for j in range(n):
                    
                    if j != i:
                        TCs[j] = upper[j]  # Reset other TCs to their upper bounds
                        
                TCs[i] = max(lower[i], TCs[i]-0.0001)  # Slightly reduce the TC being optimized
                excess_sum = np.sum(TCs) - 1  # Recalculate the excess sumt

        # Ensure that the sum of TCs is close to 1
        if np.isclose(np.sum(TCs), 1):
            
            optimized_combinations.append(TCs)  # Save the optimized combination
            
    # Extract the individual optimized TCs (diagonal elements)
    optimized_max = np.array(optimized_combinations)
    constraints_max = np.array([optimized_max[i][i] for i in range(n)])
    
    return constraints_max  # Return the optimized values

def trimming(lower, upper):
    '''
    Computes the minimum and maximum values for each TC under the sum constraint.

    Parameters:
    lower (list or array): Lower uncertainty bounds for each TC.
    upper (list or array): Upper uncertainty bounds for each TC.

    Returns:
    tuple: Two arrays containing the optimized minimum and maximum values for each TC.
    '''
    
    trimmed_lower = min_trimming(lower, upper)
    trimmed_upper = max_trimming(lower, upper)
    
    return trimmed_lower, trimmed_upper

#%% Sampling

def two_blocked_gibbs_sampling(lower, upper, e):
    """
    Uses a pairwise Gibbs sampler to sample from the set of vectors x that satisfy:
      - lower[i] <= x[i] <= upper[i] for each coordinate, and
      - sum(x) = 1
     
     Parameters:
     - lower: list or array
         Lower uncertainty bounds for each TC.
     - upper: list or array
         Upper uncertainty bounds for each TC.
    - e: int
        Number of samples desired.
    
    Returns:
    - samples: list of np.ndarray
        A list containing `e` samples of valid TC values
    """

    n = len(lower)
    
    # Check that a feasible region exists
    tolerance = 1e-4
    if np.sum(lower) > 1 + tolerance or np.sum(upper) < 1 - tolerance:
        raise ValueError(f"The feasible region is empty! sum(lower): {np.sum(lower)}, sum(upper): {np.sum(upper)}")
    
    # Initialize a valid starting point, starting at lower and distributing the remainder proportionally.
    x = np.array(lower, dtype=float)
    remainder = 1 - np.sum(x)
    gaps = np.array(upper) - np.array(lower)
    if np.sum(gaps) > 0:
        x += remainder * gaps / np.sum(gaps)
    else:
        # If no gap exists, x is already fixed.
        x = np.array(lower, dtype=float)
    
    # Ensure that the initial x is feasible.
    assert np.all(x >= lower - tolerance), "Failed lower bound"
    assert np.all(x <= upper + tolerance), "Failed upper bound"
    assert np.isclose(np.sum(x), 1, atol=tolerance), f'Sum not close: sum(x) = {np.sum(x)}'
    
    samples = []
    iteration = 0
    
    # Main Gibbs sampling loop.
    while len(samples) < e:
        iteration += 1
        # Update every pair (i, j) in a systematic order.
        for i in range(n):
            for j in range(i+1, n):
                # Let s be the total mass for components i and j.
                s = x[i] + x[j]
                # The new value for x[i] (call it new_x_i) must lie in the intersection of:
                #   [lower[i], upper[i]] and [s - upper[j], s - lower[j]]
                new_lower = max(lower[i], s - upper[j])
                new_upper = min(upper[i], s - lower[j])

                # Sample new_x_i uniformly in the valid range.
                new_x_i = np.random.uniform(new_lower, new_upper)
                new_x_j = s - new_x_i
                # Update the pair.
                x[i] = new_x_i
                x[j] = new_x_j
                
                # (At this point, x[i] and x[j] are updated, and the sum x[i] + x[j] remains s.)
                
        # Record the sample
        samples.append(np.copy(x))

    return samples

def normalized_sampling(lower, upper, e):
    """
    Uses a uniformer sampler within lower and upper bounds with normalization
     
     Parameters:
     - lower: list or array
         Lower uncertainty bounds for each TC.
     - upper: list or array
         Upper uncertainty bounds for each TC.
    - e: int
        Number of samples desired.
    
    Returns:
    - samples: list of np.ndarray
        A list containing `e` samples of mass-balanced TC values
    """
    
    n = len(lower)
    
    # Generate e samples of dimension n
    raw_samples = np.random.uniform(lower, upper, size=(e, n))
    
    # Normalize each sample to sum to 1
    normalized_samples = [sample / np.sum(sample) for sample in raw_samples]
    
    return normalized_samples
    

def generate_samples(df, e, Gibbs=True):
    """
    Generates a set of samples based on the uncertainty bounds for each 'From' group in the input DataFrame.
    For each group, the function samples values within the specified lower and upper bounds while
    respecting the sum constraint.

    Parameters:
    - df : pandas.DataFrame
        The input DataFrame containing uncertainty bounds and destination data. 
    - e : int
        The number of samples to generate for each 'From' group.

    Returns:
    - pandas.DataFrame
        A new DataFrame that includes the original bounds and the generated samples for each experiment.
    """
    
    # Group the DataFrame by the 'From' column to handle each process independently
    grouped = df.groupby('From')
    
    # Make a copy of the original DataFrame to preserve the original structure
    base_df = df.copy().reset_index(drop=True)

    # Initialize a DataFrame to hold all the generated samples
    # The index will match the original DataFrame, and the columns will be numbered [0, 1, 2, ..., e-1]
    sample_matrix = pd.DataFrame(index=base_df.index, columns=[str(i) for i in range(e)], dtype=float)

    # Iterate over each group in the 'From' column
    for group_name, group in grouped:
        
        # Extract the lower and upper bounds for the uncertain factors in the current group
        lower = group['Lower'].values
        upper = group['Upper'].values
        
        # Get the destinations corresponding to the current process
        destinations = group['To'].values
        
        # If there are multiple lower and upper bounds, use the pairwise Gibbs sampling method
        if len(lower) > 1:
            # Perform two-blocked Gibbs sampling to generate e samples
            if Gibbs == True:
                samples = two_blocked_gibbs_sampling(lower, upper, e)
            elif Gibbs == False:
                samples = normalized_sampling(lower, upper, e)
        else:
            # If there is only one lower and upper bound, sample uniformly within the range for each destination
            samples = [[sample] for sample in np.random.uniform(lower, upper, e)]

        # For each experiment (sample) generated
        for exp in range(e):
            # For each destination in the current group
            for to, value in zip(destinations, samples[exp]):
                
                # Find the index of the row in the base DataFrame that corresponds to the current 'From' and 'To' pair
                mask = (base_df['From'] == group_name) & (base_df['To'] == to)
                
                # Since we expect only one match for each (From, To) pair, we fetch its index
                idx = base_df[mask].index[0]
                
                # Insert the sampled value into the appropriate location in the sample_matrix DataFrame
                sample_matrix.at[idx, str(exp)] = value

    # Concatenate the sample values with the original DataFrame to form the final DataFrame
    full_df = pd.concat([base_df, sample_matrix], axis=1)

    return full_df

#%% Apply levers


def apply_levers(policy_levers, mfa_params, adm_params): 
    """
    Apply policy levers by multiplying current the value of a TC with lever
    value. Add new flows is the TC is not existent. Rescale remaining flows
    to ensure mass-balance.
    
    Parameters:
    - policy_levers : pandas.DataFrame
        A DataFrame containing the levers to apply. Must include columns:. 
    - mfa_params : pandas.DataFrame
        The MFA parameter table. 
    - adm_params : pandas.DataFrame
        The ADM parameter table

    Returns:
    - Tuple[pandas.DataFrame, pandas.DataFrame]
        Updated (mfa_params, adm_params) DataFrames with the levers applied.
    """
    
    # Iterate over all processes affected by levers
    lever_processes = policy_levers['source'].unique()

    for process in lever_processes:
        # Determine which module this process belongs to (mfa or adm)
        module = policy_levers[policy_levers['source'] == process]['type'].iloc[0]
        mask_process = (policy_levers['type'] == module) & (policy_levers['source'] == process)
        
        if module == 'mfa':
            lever_targets = policy_levers[mask_process]['target']
            total_lever_value = 0

            # Apply or add levers to MFA flows
            for _, row in policy_levers[mask_process].iterrows():
                mask = (mfa_params['From'] == row['source']) & (mfa_params['To'] == row['target'])
                if mask.any():
                    # Modify existing flow
                    current_value = mfa_params.loc[mask, 'PET'].values[0]
                    new_value = max(0,min(1,current_value * row['value']))
                    mfa_params.loc[mask, 'PET'] = new_value
                else:
                    # Add new flow if it does not exist
                    new_value = row['value']
                    new_row = pd.DataFrame([{'From': row['source'], 'To': row['target'], 'PET': new_value}])
                    mfa_params = pd.concat([mfa_params, new_row], ignore_index=True)
                total_lever_value += new_value

            # Rescale remaining flows (not affected by levers) to maintain total = 1
            antimask = (mfa_params['From'] == process) & (~mfa_params['To'].isin(lever_targets))
            sum_other_TCs = mfa_params.loc[antimask, 'PET'].sum()
            gap = 1.0 - total_lever_value

            if gap < 0:
                raise ValueError(f"Policy levers for '{process}' exceed 1.0 — cannot apply them without violating mass balance.")

            if sum_other_TCs > 0:
                # Proportionally scale remaining flows to fill the gap
                mfa_params.loc[antimask, 'PET'] *= gap / sum_other_TCs
            else:
                # No remaining flows: ensure all flow goes to lever targets
                mfa_params.loc[antimask, 'PET'] = 0.0

        elif module == 'adm':
            lever_targets = policy_levers[mask_process]['target']
            total_lever_value = 0

            # Apply or add levers to ADM flows
            for _, row in policy_levers[mask_process].iterrows():
                mask = (adm_params['From'] == row['source']) & (adm_params['To'] == row['target'])
                if mask.any():
                    # Modify existing degradation flow
                    current_value = adm_params.loc[mask, 'Degradation'].values[0]
                    new_value = max(0,min(100,current_value * row['value']))
                    adm_params.loc[mask, 'Degradation'] = new_value
                else:
                    # Add new degradation flow
                    new_value = row['value']
                    new_row = pd.DataFrame([{'From': row['source'], 'To': row['target'], 'Degradation': new_value}])
                    adm_params = pd.concat([adm_params, new_row], ignore_index=True)
                total_lever_value += new_value

            # Rescale remaining flows to maintain sum = 1
            antimask = (adm_params['From'] == process) & (~adm_params['To'].isin(lever_targets))
            sum_other_TCs = adm_params.loc[antimask, 'Degradation'].sum()
            gap = 1.0 - total_lever_value

            if gap < 0:
                raise ValueError(f"Policy levers for '{process}' exceed 1.0 — cannot apply them without violating mass balance.")

            if sum_other_TCs > 0:
                adm_params.loc[antimask, 'Degradation'] *= gap / sum_other_TCs
            else:
                adm_params.loc[antimask, 'Degradation'] = 0.0

    return mfa_params, adm_params

#%% Aggregate metrics

def compute_metrics(df, metrics):
    '''Computes the metrics of interest for each MFA experiment based on accumulation
    in various sinks in the final year of the MFA model, and stores these results
    in df.
    
    Parameters:
    - df : pandas.DataFrame
        The DataFrame containing MFA results
    - metrics : pandas.DataFrame
        The DataFrame specifying the metrics of interest
    '''
    
    # Iterate over policies
    for policy, experiments in df.items():
        
        # Iterate over experiments
        for experiment, values in experiments.items():
            acc = values['Accumulation']  # Load accumulation results
            
            # Iterate over metrics of interest
            for idx, row in metrics.iterrows():
                all_sinks = 0
                
                # Iterate over sinks
                for sink in metrics.columns:
                    if sink not in ['Label', 'Minimize', 'Threshold', 'Relative']:
                        
                        # If the sink should be included in the outcome, add its value to all_sinks
                        if row[sink]:
                            if sink in acc['Process'].values:
                                sink_acc = acc.loc[acc['Process'] == sink, 'Total'].values[0]
                                all_sinks += sink_acc
                                
                # Store the value of the outcome of interest in the df            
                values[idx] = all_sinks
             
    
#%% Visualization

def dotplot(df, metric, folder, policy_colors, minimize, policy_order=None):
    os.makedirs(folder, exist_ok=True)

    data = []
    for policy, experiments in df.items():
        for experiment, values in experiments.items():
            non_metrics = ['MFA_params', 'ADM_params', 'Accumulation', 'Throughput']
            metrics_dict = {k: v for k, v in values.items() if k not in non_metrics}
            if metric in metrics_dict:
                data_point = {'Policy': policy, metric: metrics_dict[metric]}
                data.append(data_point)

    df_expanded = pd.DataFrame(data)
    df_melted = df_expanded.melt(id_vars='Policy', var_name='Metric', value_name='Value')
    subset = df_melted[df_melted['Metric'] == metric]

    sns.set(style="whitegrid", palette="tab10")
    plt.figure(figsize=(7, 6))

    ax = sns.stripplot(
        data=subset,
        x='Policy',
        y='Value',
        hue='Policy',
        order=policy_order,
        palette={policy: policy_colors.get(policy, "#333333") for policy in policy_order},
        jitter=0.2,
        dodge=False,
        alpha=0.3,
        size=5
    )

    if ax.legend_ is not None:
        ax.legend_.remove()

    # Map policy names to numeric positions on x-axis
    policy_to_x = {policy: i for i, policy in enumerate(policy_order)}

    max_points = 5  # number of points to connect horizontally

    # For each point index from 0 to max_points-1,
    # connect the points of all policies horizontally if they exist.
    for i in range(max_points):
        x_vals = []
        y_vals = []
        for policy in policy_order:
            vals = subset[subset['Policy'] == policy]['Value'].values
            if len(vals) > i:  # if policy has enough points
                x_vals.append(policy_to_x[policy])
                y_vals.append(vals[i])
        if len(x_vals) > 1:
            ax.plot(x_vals, y_vals, color='black', alpha=0.3, linewidth=1)

    ax.set_title(f'{metric.replace("_", " ")}', fontsize=16)
    ax.set_ylabel('Mass (t)', fontsize=16)
    ax.set_xlabel('', fontsize=16)
    ax.set_facecolor('#fafafa')

    # Ensure 0 is on the y-axis
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(bottom=min(0, y_min), top=y_max)

    # Set x-ticks and increase font size
    ax.set_xticks(range(len(policy_order)))
    ax.set_xticklabels(policy_order, rotation=-45, ha='left', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)

    safe_metric = metric.replace("/", "-").replace(" ", "_")
    plt.tight_layout()
    plt.savefig(f'{folder}/dotplot_{safe_metric}.png')
    plt.close()
    
def custom_formatter(x, i, minimize):
    """
    Formats numerical values for plotting axes depending on the index `i` and
    whether the objective is to minimize or maximize metrics.
    
    Parameters:
    - x : float
        The value to format.
    - i : int
        Index representing the metric type
    - minimize : bool
        If True, use scientific notation for metric 2; otherwise use standard formatting.
    
    Returns:
    - str
        A formatted string representation of `x` based on the selected metric.
    """
    
    if i == 0 or i == 3 or i == 4:
        # Divide by e3 (thsnd) and format as kt (kilo tons)
        return f"{x / 1e3:,.1f} kt"
    elif i == 1:
        # Format as percentage
        return f"{x * 100:.0f}%"
    elif i == 2:
        # Format as scientific notation unless minimize = False
        if minimize == False:
            return f"{x:.3f}"  # Normal number with one decimal
        else:
            return f"{x:.2e} t$^{2}$"  # Scientific notation
    else:
        # Default formatting if none of the conditions match
        return f"{x:.2f}"
    
def parallel_coordinate(robustness, lab, minimize, folder, policy_colors,
                        custom_axis_ranges={
                            0: (0, 300000),
                            1: (0, 1),
                            2: (0, 200000000),
                            3: (0, 100000),
                            4: (0, 100000)},
                        policy_order=None):
    """
    Create and save a parallel coordinate plot for visualizing policy
    robustness across multiple metrics.

    Parameters:
    - robustness: DataFrame of metric values and policy labels
    - lab: str, used in title and filename
    - minimize: bool, indicates whether the main objective is to minimize
    - folder: str, output directory
    - policy_colors: dict mapping policy names to color codes
    - custom_axis_ranges: dict mapping metric indices to (min, max) ranges
    - policy_order: list of policies to use in legend order
    """

    # Exclude BAU policy
    robustness = robustness.copy()
    robustness = robustness[robustness['Policy'] != 'BAU']

    # Extract list of metrics (exclude Policy column)
    metrics = list(robustness.columns)
    metrics.remove('Policy')

    # Determine axis ranges per metric
    orig_min_max = {}
    for i, m in enumerate(metrics):
        if i in custom_axis_ranges:
            mn, mx = custom_axis_ranges[i]
        elif i == 1:
            mn, mx = 0, 1
        else:
            mn, mx = robustness[m].min(), robustness[m].max()
        orig_min_max[m] = (mn, mx)

    # Set up the overall figure
    dims = len(metrics)
    fig = plt.figure(figsize=(12, 6))

    left_margin = 0.1
    right_margin = 0.9
    spacing = (right_margin - left_margin) / (dims - 1)
    ax_width = 0.05
    ax_bottom = 0.15
    ax_height = 0.7

    axes = []
    
    for i in range(dims):
        pos = [left_margin + i * spacing - ax_width / 2, ax_bottom, ax_width, ax_height]
        ax = fig.add_axes(pos)
    
        mn, mx = orig_min_max[metrics[i]]
        ax.set_ylim(mn, mx)
    
        yticks = np.linspace(mn, mx, 6)
        ax.set_yticks(yticks)
    
        # Set background color
        ax.set_facecolor('#fafafa')
        
        formatter = ticker.FuncFormatter(lambda x, _: custom_formatter(x, i, minimize))
        # Set y-tick labels
        ax.set_yticklabels([formatter(tick, None) for tick in yticks], fontsize=16)
    
        ax.set_xticks([])
        ax.set_title(metrics[i], fontsize=16, fontweight='bold', pad=16)
    
        # Hide right and left spines (vertical lines)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        # Remove horizontal spines and grid to prevent gray lines
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(False)
    
        # Make sure y-axis ticks are drawn
        ax.tick_params(axis='y', which='both', length=6, direction='in')
    
        axes.append(ax)

    # Draw custom black grid lines to the left only
    for ax in axes:
        pos = ax.get_position()
        x_center = pos.x0 + pos.width / 2
        yticks = ax.get_yticks()
        for ytick in yticks:
            # Normalize y position
            mn, mx = ax.get_ylim()
            norm = (ytick - mn) / (mx - mn) if mx - mn else 0.5
            y_fig = pos.y0 + norm * pos.height
            # Draw horizontal line from left edge to axis center
            fig.lines.append(mlines.Line2D(
                [pos.x0, x_center], [y_fig, y_fig],
                color='black', linewidth=0.4,
                transform=fig.transFigure))

    # Draw vertical axis lines (behind the data lines)
    for ax in axes:
        pos = ax.get_position()
        x_center = pos.x0 + pos.width / 2
        fig.lines.append(mlines.Line2D(
            [x_center, x_center], [pos.y0, pos.y0 + pos.height],
            color='black', linewidth=0.6,
            transform=fig.transFigure))
        
    for i in range(dims - 1):
        right_edge = left_margin + i * spacing + ax_width / 2
        left_edge_next = left_margin + (i + 1) * spacing - ax_width / 2
        gap_left = right_edge
        gap_width = left_edge_next - right_edge
        rect = patches.Rectangle(
            (gap_left, ax_bottom),
            gap_width,
            ax_height,
            transform=fig.transFigure,
            color='#fafafa',
            zorder=-1
        )
        fig.patches.append(rect)
        
    # Draw each policy's line across the axes
    for _, row in robustness.iterrows():
        pts = []
        for i, m in enumerate(metrics):
            ax = axes[i]
            pos = ax.get_position()
            mn, mx = orig_min_max[m]
            val = row[m]
            norm = (val - mn) / (mx - mn) if mx - mn else 0.5
            y_fig = pos.y0 + norm * pos.height
            x_fig = pos.x0 + pos.width / 2
            pts.append((x_fig, y_fig))
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            fig.lines.append(
                mlines.Line2D(
                    [x0, x1], [y0, y1],
                    color=policy_colors[row['Policy']],
                    linewidth=2,
                    transform=fig.transFigure,
                    alpha=0.6
                )
            )

    # Create custom legend in user-specified order
    if policy_order is not None:
        legend_handles = [mlines.Line2D([], [], color=policy_colors[policy], lw=2, label=policy)
                          for policy in policy_order]

    # Add plot title
    plt.text(0.5, 0.95, f'Outcome of interest: {"minimizing" if minimize else "maximizing"} {lab.replace("_", " ")}',
             ha='center', va='bottom', fontsize=16, transform=fig.transFigure)

    # Draw legend
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=16
    )

    # Final layout and save
    plt.subplots_adjust(top=0.85, bottom=0.25)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/robustness_{lab}')
    plt.close()
    
def Sankey(policy, df, lab, minimize, spearman_corr, folder, ref_year, final_year):
    """
    Generates and saves a dual Sankey diagram comparing the worst-case and best-case scenarios
    for PET plastic accumulation across metrics of interest, highlighting vulnerable flows.

    Parameters:
    - policy: str
        Identifier for the policy being analyzed.
    - df: dict
        Dictionary of model experiment results. Each key maps to a nested structure 
        with 'Throughput' and 'MFA_params' DataFrames.
    - lab: str
        The label of the outcome variable of interest (e.g., accumulation compartment).
    - minimize: bool
        Whether the objective is to minimize the outcome of interest.
    - spearman_corr: dict
        Dictionary containing 'correlation' and 'p-values' DataFrames
        with Spearman correlation results between model parameters and flows.
    - folder: str
        Path to the folder where the output HTML Sankey diagram should be saved.
        ref_year (int): Reference year for the scenario being visualized.
    - ref_year : int
        year for which the sankey diagram is plotted
    - final_year : int
        final year setting the temporal scope for when best- and worst-cases are determined

    Output:
    - Saves an interactive HTML Sankey diagram comparing best- and worst-case scenarios.
    """

    # Identify experiments with worst and best metrics based on the lab variable
    exp_worst = max(df, key=lambda k: df[k][lab])
    exp_best = min(df, key=lambda k: df[k][lab])

    # Flip if we're maximizing instead of minimizing
    if not minimize:
        exp_worst, exp_best = exp_best, exp_worst

    # Extract unique process names to use as nodes
    nodes = np.unique(df[0]['Throughput']['Process'].values)
    num_nodes = len(nodes)

    node_colors = ['white'] * num_nodes  # Default node color

    Sankeys = []

    corr = spearman_corr['correlation']
    pvals = spearman_corr['p-values']

    for exp in [exp_worst, exp_best]:
        A = df[exp]['MFA_params']   # Transfer coefficients
        X = df[exp]['Throughput']   # Throughput values

        # Set color map: Red for positive influence, Blue for negative
        colorscale = 'RdBu' if minimize else 'RdBu_r'
        cmap = cm.get_cmap(colorscale)

        links = []
        for source in nodes:
            for target in nodes:
                if ((A['From'] == source) & (A['To'] == target)).any():
                    TC = A.loc[(A['From'] == source) & (A['To'] == target), 'PET'].iloc[0]
                    inflow = X.loc[X['Process'] == source, 'Packaging'].iloc[0]
                    flow = TC * inflow
                    if flow != 0:
                        source_idx = np.where(nodes == source)[0][0]
                        target_idx = np.where(nodes == target)[0][0]
                        rho = corr.loc[(corr['From'] == source) & (corr['To'] == target), 'correlation'].values[0]
                        
                        # Full Sankey
                        pval = pvals.loc[(pvals['From'] == source) & (pvals['To'] == target), 'p-value'].values[0]
                        
                        if pval < 0.05:
                            color = rho # Base flow color on correlation
                        else:
                            color = 0 # Do not color if statistically insignificant
                            
                        links.append([source_idx, target_idx, flow, rho, color, f'rho: {round(rho, 2)}, pval: {round(pval, 2)}'])

        # Unpack Sankey link parameters
        sources, targets, flows, rho_values, color_values, rhos = zip(*links)
        
        # Get colors
        norm = Normalize(vmin=-1, vmax=1)
        colors = [to_hex(cmap(norm(color))) for color in color_values]

        Sankey_diagram = go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=node_colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=flows,
                color=colors,
                label=rhos
            )
        )

        Sankeys.append(Sankey_diagram)

    # Create two side-by-side Sankey diagrams
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "sankey"}, {"type": "sankey"}],[{"type": "xy"},{"type": "xy"}]],
        row_heights=[1.2,0.001],
        column_widths=[0.4, 0.4]  # allocate less space for colorbar
    )

    fig.add_trace(Sankeys[0], row=1, col=1)
    fig.add_trace(Sankeys[1], row=1, col=2)

    # Adjust subplot title positions
    for annotation in fig['layout']['annotations']:
        annotation['y'] -= 0.1

    # Create a linear gradient of values for the colorbar (e.g. 100 steps)
    colorbar_vals = np.linspace(-1, 1, 100)

    # Add invisible scatter trace for colorbar
    colorbar =  go.Scatter(
        x=colorbar_vals,   # x varies from -cscale to cscale (horizontal axis)
        y=[0]*100,        # y fixed at 0 (horizontal strip)
        mode='markers',
        marker=dict(
            color=colorbar_vals,
            colorscale=colorscale,
            cmin=-1,
            cmax=1,
            size=0,
            showscale=True,
            colorbar=dict(
                title="Spearman's rank correlation coefficient",
                titleside="top",
                ticks="outside",
                tickvals=[-1, 0, 1],
                ticktext=[f"-{1:.2f}", "0", f"{1:.2f}"],
                len=0.2,
                x=0.5,
                xanchor="center",
                y=-0.2,            # position below the markers
                yanchor="top",
                orientation='h'    # horizontal orientation
            )
        ),
        hoverinfo='none',
        showlegend=False)

    fig.add_trace(colorbar, row=2, col=1)
    
    # Hide axes for the colorbar subplot
    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(visible=True, row=2, col=1)


    # Update layout with title and figure size
    fig.update_layout(
        height=1500,
        width=6000,
        font_size=30,
        font_color="black")

    # Save the Sankey diagram as an interactive HTML file
    fig.write_html(f'{folder}/sankey_{policy}_{lab}.html')

    
#%% Robustness evaluation

def mean_outcome(df, lab):
    """
    Calculates the mean outcome per policy for a specific outcome label.

    - df: dict
        Dictionary where keys are policy indices and values are dictionaries of experiments.
    - lab: str
        The outcome label to evaluate.

    Returns:
    - dict
        Nested dictionary with mean outcome values for each policy under the specified label.
    """
    
    mean_metrics = {}

    mean_metrics[lab] = {}
    for policy, experiments in df.items():
        # Calculate mean outcome for each policy for the given lab
        mean_metrics[lab][policy] = np.mean([exp[lab] for exp in experiments.values()])
        
    return mean_metrics

def right_skewness(df, lab):
    """
    Calculates the right skewness of metrics per policy.

    - df: dict
        Dictionary where keys are policy indices and values are dictionaries of experiments.
    - lab: str
        The outcome label to evaluate.

    Returns:
    - dict
        Nested dictionary of right skewness values per policy.
    """
    
    right_skewness = {}

    right_skewness[lab] = {}
    for policy, experiments in df.items():
        # Calculate right skewness for each policy for the given lab
        metrics = [exp[lab] for exp in experiments.values()]
        p10 = np.percentile(metrics, 10)
        p50 = np.percentile(metrics, 50)
        p90 = np.percentile(metrics, 90)
        right_skewness[lab][policy] = ((p90 + p10) / (2 - p50)) / ((p90 - p10) / 2)
            
    return right_skewness

def domain_criterion(df, threshold, lab):
    """
    Computes the domain criterion (failure or success rate) for each policy, 
    based on whether metrics exceed a given threshold.
    
    - df: dict
        Dictionary of policy experiments.
    - threshold: float
        Threshold value to determine failure or success.
    - lab: str
        Outcome label to assess.
    
    Returns:
    - dict
        Nested dictionary with proportion of experiments failing (or succeeding) for each policy.
    """
    
    domain_crit = {}

    domain_crit[lab] = {}
    for policy, experiments in df.items():
        # Calculate domain criterion (percentage of failures) for each policy for the given lab
        domain = sum(1 for exp in experiments.values() if (exp[lab] > threshold).sum() > 0)
        domain_crit[lab][policy] = domain / len(experiments) if len(experiments) > 0 else 0.0

    return domain_crit

def signal_x_noise(df, lab):
    """
    Computes the product of signal (mean) and noise (standard deviation) plus one,
    used as a robustness indicator for minimizing objectives.

    - df: dict
        Dictionary of experiments by policy.
    - lab: str
        The outcome label.

    Returns:
    - dict
        Nested dictionary with signal x noise values for each policy.
    """
    
    sxn = {}

    sxn[lab] = {}
    for policy, experiments in df.items():
        # Calculate signal x noise for each policy for the given lab
        metrics = [exp[lab] for exp in experiments.values()]
        signal = np.mean(metrics)
        noise = np.std(metrics)
        sxn[lab][policy] = (signal + 1) * (noise + 1)
    
    return sxn

def max_outcome(df, lab):
    """
    Returns the maximum outcome value across all experiments per policy.

    - df: dict
        Dictionary of policy experiments.
    - lab: str
        The outcome label to evaluate.

    Returns:
    - dict
        Nested dictionary with maximum outcome per policy.
    """
    
    max_metrics = {}

    max_metrics[lab] = {}
    for policy, experiments in df.items():
        # Calculate max outcome for each policy for the given lab
        max_metrics[lab][policy] = np.max([exp[lab] for exp in experiments.values()])
        
    return max_metrics

def signal_to_noise(df, lab):
    """
    Computes the signal-to-noise ratio (mean divided by standard deviation, both plus one)
    as a robustness indicator for maximizing objectives.

    - df: dict
        Dictionary of policy experiments.
    - lab: str
        Outcome label.

    Returns:
    - dict
        Nested dictionary with signal-to-noise ratio values for each policy.
    """
    
    s2n = {}

    s2n[lab] = {}
    for policy, experiments in df.items():
        # Calculate signal to noise for each policy for the given lab
        metrics = [exp[lab] for exp in experiments.values()]
        signal = np.mean(metrics)
        noise = np.std(metrics)
        s2n[lab][policy] = (signal + 1) / (noise + 1)
    
    return s2n

def min_outcome(df, lab):
    """
    Calculates the minimum outcome across all experiments per policy.

    - df: dict
        Dictionary of policy experiments.
    - lab: str
        The outcome label to evaluate.

    Returns:
    - dict
        Nested dictionary with minimum outcome values per policy.
    """
    
    min_metrics = {}

    min_metrics[lab] = {}
    for policy, experiments in df.items():
        # Calculate min outcome for each policy for the given lab
        metrics = [exp[lab] for exp in experiments.values()]
        min_metrics[lab][policy] = np.min(metrics)
        
    return min_metrics

def regret_margin(df, lab):
    """
    Calculates the distance from the max regret to the max possible regret.

    Parameters:
    - df: dict
        Dictionary of policy experiments.
    - lab: str
        The outcome label to evaluate.

    Returns:
    - dict
        Nested dictionary with regret margin values per policy.
    """
    
    policies = list(df.keys())
    states = df[policies[0]].keys()  # assumes all policies cover the same states

    max_regret_result = {lab: {policy: 0 for policy in policies}}

    for state in states:
        # Find the best (maximum) outcome across policies for this state
        best_value = max(df[policy][state][lab] for policy in policies)

        # Compute regret for each policy in this state
        for policy in policies:
            regret = best_value - df[policy][state][lab]
            max_regret_result[lab][policy] = max(max_regret_result[lab][policy], regret)
            
    max_pos_regret = max([max_regret_result[lab][policy] for policy in policies])
    regret_margin_result = {lab: {policy: max_pos_regret - max_regret_result[lab][policy] for policy in policies}}
        
    return regret_margin_result

def max_regret(df, lab):
    """
    Calculates the maximum regret per policy across all states of the world.

    Parameters:
    - df: dict
        Dictionary of policy experiments.
    - lab: str
        The outcome label to evaluate.

    Returns:
    - dict
        Nested dictionary with max regret values per policy.
    """
    
    policies = list(df.keys())
    states = df[policies[0]].keys()  # assumes all policies cover the same states

    max_regret_result = {lab: {policy: 0 for policy in policies}}

    for state in states:
        # Find the best (minimum) outcome across policies for this state
        best_value = min(df[policy][state][lab] for policy in policies)

        # Compute regret for each policy in this state
        for policy in policies:
            regret = df[policy][state][lab] - best_value
            max_regret_result[lab][policy] = max(max_regret_result[lab][policy], regret)

    return max_regret_result
    
def robustness_evaluation(df, metrics):
    """
    Evaluates robustness metrics across policies for all defined metrics,
    based on whether they should be minimized or maximized.

    - df: dict
        Dictionary of experiments grouped by policy, with nested outcome DataFrames.
    - metrics: pd.DataFrame
        DataFrame where rows are outcome labels and columns include:
        'Minimize' (bool), 'Threshold' (float), and 'Relative' ('TOT', 'BAU', or other).

    Returns:
    - dict
        Dictionary of DataFrames, each containing metrics for a specific outcome across policies.
    """
    
    metrics_df = {}
    
    # Calculate total average accumulation across all experiments (used for relative thresholds)
    tot = sum(df['BAU'][x]['Accumulation']['Total'].sum() for x in df['BAU']) / len(df['BAU'])
    
    # Get outcome labels (objectives) and their corresponding minimization flags
    outcome_labels = metrics.index
    minimize_flags = metrics['Minimize']
    
    # Loop through each objective to compute robustness metrics
    for lab, flag in zip(outcome_labels, minimize_flags):
        metrics_df[lab] = {}
    
        # Retrieve the threshold and its relative type (TOT or BAU)
        threshold = metrics.loc[metrics.index == lab, 'Threshold']
        relative = metrics.loc[metrics.index == lab, 'Relative'].iloc[0]
    
        # Convert relative threshold to absolute value
        if relative == 'TOT':
            threshold = threshold * tot
        elif relative == 'BAU':
            bau = sum(df['BAU'''][x][lab] for x in df['BAU']) / len(df['BAU'])
            threshold = threshold * bau
    
        # Compute metrics common to all objectives
        metrics_df[lab]['Mean Outcome'] = mean_outcome(df, lab)
    
        # Compute additional metrics depending on whether the objective should be minimized or maximized
        if flag:  # Objective should be minimized
            metrics_df[lab]['Failure Rate'] = domain_criterion(df, threshold, lab)
            metrics_df[lab]['Signal X Noise'] = signal_x_noise(df, lab)
            metrics_df[lab]['Max Outcome'] = max_outcome(df, lab)
            metrics_df[lab]['Max Regret'] = max_regret(df, lab)
        else:     # Objective should be maximized
            metrics_df[lab]['Success Rate'] = domain_criterion(df, threshold, lab)
            metrics_df[lab]['Signal to Noise'] = signal_to_noise(df, lab)
            metrics_df[lab]['Min Outcome'] = min_outcome(df, lab)
            metrics_df[lab]['Regret margin'] = regret_margin(df, lab)
    
    # Format each objective's results into a clean DataFrame
    for lab in outcome_labels:
        data = {}
    
        # Extract just the metric values for each policy
        for metric, values in metrics_df[lab].items():
            data[metric] = [list(inner_dict.values()) for inner_dict in values.values()][0]
    
        # Create a DataFrame indexed by policy IDs
        policy_names = list(df.keys())
        df_lab = pd.DataFrame(data, index=policy_names)
    
        # Reset index and label policies for clarity (e.g. "No Policy", "Policy 1", ...)
        df_lab = df_lab.reset_index()
        df_lab['Policy'] = df_lab['index']
        df_lab = df_lab.drop(columns='index')
    
        # Replace raw metrics dictionary with formatted DataFrame
        metrics_df[lab] = df_lab
    
    return metrics_df

#%% Vulnerability analysis

def spearman(df, metrics):
    """
    Calculate Spearman rank correlation between transfer coefficients and metrics
    for each policy.

    - var: df : dict
      Dictionary of the form {policy: {experiment: {'MFA_params': DataFrame, <outcome_name>: float}}}
      containing MFA parameters and outcome values for each experiment under each policy.

    - var: metrics : pandas.DataFrame
      DataFrame with outcome names as index, typically containing objective-specific metadata.

    Returns:
    - var: spearman_indices : dict
      Nested dictionary structured as:
      {policy: {outcome_name: {'correlation': DataFrame, 'p-values': DataFrame}}}
      containing Spearman correlation coefficients and associated p-values for each (From, To) link.
    """
    
    spearman_indices = {}

    policies = df.keys()  # Number of policies
    e = len(df['BAU'].items())  # Number of experiments per policy

    # Loop over each outcome of interest
    for outcome_name in metrics.index:

        # Loop over each policy
        for policy in policies:

            # Initialize nested dictionary to store results per policy and outcome
            if policy not in spearman_indices:
                spearman_indices[policy] = {}

            spearman_indices[policy][outcome_name] = {
                'correlation': pd.DataFrame(),
                'p-values': pd.DataFrame()
            }

            param_list = []       # Stores MFA_params DataFrames from each experiment
            all_metrics = []     # Stores corresponding outcome values

            # Gather parameters and metrics across experiments
            for exp in range(e):
                mfa_params = df[policy][exp]['MFA_params'].copy()
                param_list.append(mfa_params)

                accumulation = df[policy][exp][outcome_name]
                all_metrics.append(accumulation)

            # Identify all unique (From, To) link combinations across all experiments
            all_combinations = pd.concat(param_list)[['From', 'To']].drop_duplicates()
            param_names = [tuple(x) for x in all_combinations.to_numpy()]

            correlations = []  # Holds (From, To, rho) tuples
            p_values = []      # Holds (From, To, pval) tuples

            # Compute correlation for each unique (From, To) link
            for from_to in param_names:
                # Collect parameter values across all experiments for this (From, To) link
                parameter_values = [
                    df_exp.loc[
                        (df_exp['From'] == from_to[0]) & (df_exp['To'] == from_to[1]),
                        'PET'
                    ].values[0]
                    for df_exp in param_list
                ]

                # If all parameter values or metrics are constant, set correlation to 0 and p-value to NaN
                if len(set(parameter_values)) == 1 or len(set(all_metrics)) == 1:
                    rho, pval = 0, np.nan
                else:
                    # Calculate Spearman rank correlation and p-value
                    rho, pval = spearmanr(parameter_values, all_metrics)

                correlations.append((from_to[0], from_to[1], rho))
                p_values.append((from_to[0], from_to[1], pval))

            # Convert correlation results to DataFrames
            spearman_indices[policy][outcome_name]['correlation'] = pd.DataFrame(
                correlations, columns=['From', 'To', 'correlation']
            )
            spearman_indices[policy][outcome_name]['p-values'] = pd.DataFrame(
                p_values, columns=['From', 'To', 'p-value']
            )

    return spearman_indices