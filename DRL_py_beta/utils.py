"""
This file contains functions which provide some general utility functions for working with the DRL directory
"""
from glob import glob
from os.path import isdir
from os import makedirs

import torch

import numpy as np

import pandas as pd

def get_snapshot_List(simulation_re=100):
    """
    Returns a list of snapshots for a simulation with a certain reynoldsnumber
    Args:
        simulation_re: The reynoldsnumber (Re) of th simulated flow. Default set to Re=100
        start with, since the baseline data is organized by Re (e.g. Re=100)
    Returns:
        snapshotList: List of snapshots
    """
    # Make sure simultionRe is a string
    if not isinstance(simulation_re, str):
        simulation_re = str(simulation_re)
    # Get a list of available baseline data snapshots belonging to a certain reynolds number
                #DRL_py_beta/env/base_case/baseline_data/Re_{simulation_re}/processor0

    baseline_path = f'./env/base_case/baseline_data/Re_{simulation_re}/processor0/*/'
    snapshotList = glob(baseline_path, recursive=True)
    # Check if list contains something and raise exception if it is empty
    if not snapshotList:
        raise ValueError("The snapshot list is empty.")
    # Keep only the part of the strings containing the physical simulatio time
    snapshotList = [path.split('/')[-2] for path in snapshotList]
    snapshotList.remove("constant")
    snapshotList = sorted(snapshotList)
    return snapshotList

def get_random_control_start_time(simulation_re=100, lowerControlThreshold=None, upperControlThreshold=None):
    """
    Returns a random start time which is drawn from the base line data snapshots. Time boundaries can
    be set if necessary. The boundaries are not inclusive.
    Args:
        simulation_re: Contains the reynolds number (Re) of th simulated flow. Default set to Re=100
        lowerControlThreshold: Contains the lower time threshold for when to start control
        upperControlThreshold: Contains the upper time threshold for when to start control
    Returns:
        startTime: Returns a start time corresponding to the randomly selected index
        index: Returns the index of the randomly chosen point in time
    """
    # Get baseline data snapshots for given reynolds number
    snapshotList = get_snapshot_List(simulation_re)
    snapshotList = [float(snapshot) for snapshot in snapshotList]
    # Remove snapshots from list if thresholds apply
    if (lowerControlThreshold is not None) and (upperControlThreshold is None):
        new_snapshotList = [snapshot for snapshot in snapshotList if snapshot > lowerControlThreshold]
    elif (lowerControlThreshold is None) and (upperControlThreshold is not None):
        new_snapshotList = [snapshot for snapshot in snapshotList if snapshot < upperControlThreshold]
    elif (lowerControlThreshold is not None) and (upperControlThreshold is not None):
        new_snapshotList = [snapshot for snapshot in snapshotList if (snapshot > lowerControlThreshold) and (snapshot < upperControlThreshold)]
    else:
        new_snapshotList = snapshotList

    n_snapshots = len(new_snapshotList)
    # Seed the random number generator for reproducibility
    torch.manual_seed(0)
    # Draw randomly from snapshots
    index = torch.multinomial(torch.ones(n_snapshots), 1)

    startTime = new_snapshotList[index]

    return startTime, index

def cat_prediction_feature_vector(p_states, actions):
    """Concatenate states to an extended feature vector

    Parameters
    ----------
    p_states : list
        States containing the pressure values
    actions : list
        Contains action values per state

    Returns
    -------
    torch.Tensor
        Concatenated feature tensor
    """
    
    # Initialize feature vector
    feature_vector = np.zeros(shape=(p_states.shape[0]*(p_states.shape[1]+1)))
    for n, state in enumerate(p_states):
        n_state = np.append(state, [actions[n]], axis=0)
        # Insert values at the corresponding indexes
        start_idx = n*(p_states.shape[1]+1)
        end_idx = (n+1)*(p_states.shape[1]+1)
        feature_vector[start_idx:end_idx] = n_state
        
    return feature_vector

def write_model_generated_trajectory_data_to_file(n_sample: int,
                                                  n_trajectory: int,
                                                  t_data: np.array,
                                                  p_data: np.array,
                                                  c_d_data: np.array,
                                                  c_l_data: np.array,
                                                  omega_data: np.array):
    header = ["t", "c_d", "c_l", "omega"] + [f"p{n}" for n in range(1,401)]
    data = np.concatenate((t_data, c_d_data, c_l_data, omega_data, p_data), axis=1)
    data_df = pd.DataFrame(data, columns=header)
    
    file_dir = f"./Data/model_related/sample_{n_sample}/trajectory_{n_trajectory}/"
    if not isdir(file_dir):
        makedirs(file_dir)
    with open((file_dir+"trajectory.csv"), 'w') as file:
        file.write("# t and omega data is not model generated by the environment model\n")
    data_df.to_csv((file_dir+"trajectory.csv"), sep=",", index=False, mode="a")
