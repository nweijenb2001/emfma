"""
Exploratory Material Flow Modeling and Analysis (EMFMA)
    
Case study on the MFA model developed by Schwarz et al. (2023), adressing
plastic pollution from consumer PET bottles in the Netherlands.

Author: Weijenberg, N.
Institutions: Leiden University, TU Delft, TNO
Created: July 7, 2025

Note: Although the full model is not included, intermediate results
are available, allowing the data to be explored and plots to be generated.

This Python script...
(1) Loads data from the expert elicitation survey to establish plausible
uncertainty bounds for transfer coefficients (TCs) in the MFA system;
(2) Loads policy levers that define how different policies should be
implemented in the MFA model, as well as performance metrics that define how
the outcomes of the MFA should be measured;
(3) Generates states of the world by sampling within the bounds of TCs,
employing two-blocked Gibbs sampling to ensure mass-balance;
(4) Evaluates the MFA model for the Business-As-Usual (BAU) policy and
alternative policies across states of the world;
(5) Employs several robustness metrics to assess policy robustness;
(6) Employs Spearman's Rank Correlation Coefficient to identify critical TCs
determining policy performance, including a statistical test;
(7) Creates multiple figures to facilitate the interpretation of the results.
"""

#%% Define parameters and import modules

# Parameters for EMFMA
nsow = 500 # Number of states of the world to generate
Gibbs = True # True: Gibbs sampling, False: uniform sampling + normalization
newrun = True  # True: run full analysis; False: import case study results
alt_order = ['Reduce', 'Capture', 'Clean', 'Return', 'Homogenize',
                'Manage', 'Redesign', 'Extract'] # Specify plotting order
policy_order = ['BAU'] + alt_order

# Parameters for the plMFA object
scenario = 'Bau' # Must be one of 'Bau', 'Zero', or 'Reduction'
extra_mitigation_strats = None # Must be None or a list of sets
init_year = 2025 # Initial year where accumulation starts (plotted in Sankey)
final_year = 2030 # The year that the accumulation goes up to
run_identifier = 1 # Unique integer identifying the plMFA object
growsave = False # Whether or not to store the results in Parquet file format

# Importing modules
import os # For interacting with the operating system
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Set working directory
from tqdm import tqdm # For printing progress bars to the console
import pandas as pd # For structuring data
import numpy as np # For numerical operations, arrays, and functions
import modules.EMFMA as emfma # For applying EMFMA procedures
import pickle # For simple storing and loading of data structures
from matplotlib import colormaps
import warnings # For supressing warnings

# Try importing the MFA model. If not found, load old results and analyze only.
try:
    from modules.plmfa_run import main  
except ModuleNotFoundError:
    print("Module 'modules.plmfa_run' not found. Skipping run and using old results.")
    newrun = False
    
# Directory setup
dfolder = 'data' # Choose folder for storing and loading EMFMA data
rfolder = 'results' # Choose folder for results (visualizations)
os.makedirs(dfolder, exist_ok=True)  # Create data folder if nonexistent
os.makedirs(rfolder, exist_ok=True)  # Create results folder if nonexistent
warnings.simplefilter(action='ignore', category=FutureWarning) # Suppress FutureWarnings
last_status_length = 0 # Initialize status length for message updating

# File paths for reloading case study results (when newrun = False)
df_path = os.path.join(rfolder, 'df.pkl')
robustness_path = os.path.join(rfolder, 'robustness.pkl')
vulnerability_path = os.path.join(rfolder, 'vulnerability.pkl')

#%% Define helper functions

def status(msg):
    '''Prints a status update 'msg' to the console.'''
    
    global last_status_length  # Access previous status length
    padded_msg = msg.ljust(last_status_length)  # Add spaces to overwrite previous message
    print(f'\r{padded_msg}', end='', flush=True)
    last_status_length = len(msg)  # Update the status length

def run_policy_scenario(sow, policy):
    '''Runs the plMFA model for a given state of the world and policy.'''
    
    global mfa_tcs, degradation_tcs, levers, plmfa_obj, init_year, final_year

    # Extract MFA and ADM parameters for the state of the world
    mfa_params = mfa_tcs.iloc[:, [0, 1, 2, 5 + sow]].copy()
    adm_params = degradation_tcs.iloc[:, [0, 1, 2, 5 + sow]].copy()
    mfa_params.columns.values[-1] = 'PET'
    adm_params.columns.values[-1] = 'Degradation'

    # Apply policy levers to the parameters if applicable
    if policy != 'BAU':
        policy_levers = levers.loc[levers['policy'] == policy]
        mfa_params, adm_params = emfma.apply_levers(policy_levers, mfa_params, adm_params)

    # Run the plMFA model
    sink_timeseries, throughput = plmfa_obj.ema_run(mfa_params, adm_params)

    # Compute accumulation between initial and final year
    accumulation_final = sink_timeseries[sink_timeseries.index == final_year]
    accumulation_initial = sink_timeseries[sink_timeseries.index == init_year]
    accumulation_total_diff = accumulation_final['Total'].to_numpy() - accumulation_initial['Total'].to_numpy()

    # Store the accumulation difference in its own dataframe
    accumulation = accumulation_final.copy()
    accumulation['Total'] = accumulation_total_diff
    accumulation = accumulation[~np.isclose(accumulation['Total'], 0)]

    return mfa_params, adm_params, throughput, accumulation

def load_levers_and_metrics(dfolder):
    '''Loads policy lever and performance metric data.'''
    
    # Load policy lever data
    levers = pd.read_csv(os.path.join(dfolder, 'levers.csv'), header=0)
    
    # Extract the policy names
    policies = list(levers['policy'].unique())
    npol= len(policies) + 1 # Get the number of policies
    
    # Define the colormap and get tab10 colors as a list
    tab10_colors = list(colormaps['tab10'].colors)
    
    # Extract the 7th color (index 6) for 'BAU' and remove it from the list
    bau_color = tab10_colors.pop(7)
    
    # Assign the 'BAU' color first
    policy_colors = {'BAU': bau_color}
    
    # Assign the remaining colors to the other policies in order
    for policy, color in zip(policies, tab10_colors):
        policy_colors[policy] = color
    
    # Get the dataframe of performance metrics from the csv file in the dfolder
    metrics = pd.read_csv(os.path.join(dfolder, 'performance_metrics.csv'),
                            index_col=0) # Set the first column as the index
    
    # Determine the number of performance metrics
    nmet = len(metrics) 
    
    return levers, policies, npol, policy_colors, metrics, nmet

#%% Load existing results (if newrun = False)

# If conducting the analysis based on old results:
if not newrun:
    
    # Load the old results:
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
    with open(robustness_path, 'rb') as f:
        robustness = pickle.load(f)
    with open(vulnerability_path, 'rb') as f:
        vulnerability = pickle.load(f)
    
    # Load policy levers and performance metrics
    levers, policies, npol, policy_colors, metrics, nmet = load_levers_and_metrics(dfolder)

#%% Initialize the plMFA model (if newrun = True)

# If conducting the full analysis from start to end
elif newrun:
    
    # Update progress status
    status('Initializing plMFA model...')
    
    # Define MFA object used to run all scenarios
    plmfa_obj = main(scenario, extra_mitigation_strats, init_year,
                      final_year, run_identifier, growsave=growsave)
    
    #%% Manage expert elicitation
    
    # Update progress status
    status('Loading and processing data...')
        
    # Initialize dictionary to store elicitation data
    el_df = {}
    
    # Load elicitation data from the dfolder
    el_df = emfma.load_elicitation(el_df, dfolder, 'elicitation')
    
    # If there are multiple responses, aggregate data and apply trimming
    merged_el_df = emfma.merge_and_trim_elicitation(el_df)
    
    # Illustrate processed elicitation data
    emfma.plot_elicitation(merged_el_df, rfolder)
    
    # Restructure and store plausible uncertainty bounds in dfolder
    emfma.store_elicitation(merged_el_df, dfolder)
    
    #%% Load policy levers and performance metrics
    
    levers, policies, npol, policy_colors, metrics, nmet = load_levers_and_metrics(dfolder)
    
    #%% Generate states of the world
    
    # Initialize a nested dictionary for uncertainty bounds, and load data
    bounds = {'BAU':
              {'MFA':pd.read_csv(os.path.join(dfolder, 'mfa_x.csv'),
                                 index_col=[0]),
              'ADM':pd.read_csv(os.path.join(dfolder, 'adm_x.csv'),
                                 index_col=[0])}}
    
    # Generate states of the world using the plausible uncertainty bounds
    mfa_tcs = emfma.generate_samples(bounds['BAU']['MFA'], nsow, Gibbs)
    degradation_tcs = emfma.generate_samples(bounds['BAU']['ADM'], nsow, Gibbs)
    
    #%% Evaluate policies across states of the world
    
    # Initialize a nested dictionary for storing scenarios
    df = {policy:
          {exp: {'MFA_params': None, # Parameters for the MFA module
                  'ADM_params': None, # Parameters for the ADM module
                  'Throughput': None, # Throughput per process in initial year
                  'Accumulation': None} # Accumulation in sinks in final year
            for exp in range(nsow)} # Store parameters per state of the world
          for policy in ['BAU'] + policies} # And per policy
    
    # Create progress bar for running scenarios
    progress_bar = tqdm(total=len(df.keys())*nsow, desc="Running scenarios...", leave=False)
    
    # Iterate over policies
    for policy in ['BAU'] + policies:
        
        # Iterate over states of the world
        for sow in range(nsow):
            
            # Run the scenarios and store the results
            mfa_params, adm_params, throughput, accumulation = run_policy_scenario(sow, policy)
            df[policy][sow]['MFA_params'] = mfa_params
            df[policy][sow]['ADM_params'] = adm_params
            df[policy][sow]['Throughput'] = throughput
            df[policy][sow]['Accumulation'] = accumulation
            
            progress_bar.update(1) # Update the progress bar
    
    # Close the progress bar when finished
    progress_bar.close()
    
    #%% Analyze the MFA scenarios
    
    # Update progress status
    status('Analyzing scenarios...')
    
    # Compute policy performance in each scenario
    emfma.compute_metrics(df, metrics)
    
    # Evaluate the robustness of policies (aggregating performance)
    robustness = emfma.robustness_evaluation(df, metrics)
    
    # Perform vulnerability analysis to identify critical variables
    vulnerability = emfma.spearman(df, metrics)

#%% Construct figures
    
# Create progress bar for constructing figures
progress_bar = tqdm(total=nmet*2+(nmet*npol), desc="Generating figures...", leave=False)

# Define custom axis ranges for parallel coordinate plots per metric
custom_axis_ranges={'microplastics': {0: (45000, 60000),
                                      1: (0.60, 0.80),
                                      2: (900000000, 1150000000),
                                      3: (105000, 130000),
                                      4: (4000, 18000)},
                    'macroplastics':{0: (10000, 20000),
                                     1: (0.40, 0.90),
                                     2: (100000000, 200000000),
                                     3: (45000, 70000),
                                     4: (0, 25000)},
                    'secondary_material':{0: (250000, 300000),
                                          1: (0.60, 0.80),
                                          2: (2.320, 2.345),
                                          3: (40000, 60000),
                                          4: (0, 100000)}}

# Iterate over performance metrics
for metric in metrics.index:

    # Establish whether the objective is to minimize or maximize each metric
    minimize = metrics.loc[metric, 'Minimize']
    
    # # Generate a dotplot to show the distribution of policy performance
    emfma.dotplot(df, metric, rfolder, policy_colors, minimize, policy_order)
    
    # Update the progress bar
    progress_bar.update(1)
    
    # Create a parallel plot for each metric to illustrate robustness
    if metric in custom_axis_ranges.keys():
        emfma.parallel_coordinate(robustness[metric], metric, minimize, rfolder, policy_colors,custom_axis_ranges[metric], alt_order)
    
    # Use default axis ranges if the metric is undefined
    else:
        emfma.parallel_coordinate(robustness[metric], metric, minimize, rfolder, policy_colors,policy_order=alt_order)
    
    # Update the progress bar
    progress_bar.update(1)
    
    # Create a Sankey diagram for the best and worst policy outcome in init_year
    for policy in ['BAU']: # + policies:
        
        # Extract the results of vulnerability analysis
        corr = vulnerability[policy][metric] 
        
        # Create the Sankey diagram
        emfma.Sankey(policy, df[policy], metric, minimize, corr, rfolder, init_year, final_year)
        
        # Update the progress bar
        progress_bar.update(1)
    
# Close the progress bar when finished
progress_bar.close()

status('Complete')