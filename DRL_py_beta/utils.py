"""
This file contains the utils class which provides some general utility functions for working with the DRL directory
"""
from glob import glob

import torch


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

def cat_prediction_feature_vector(p_states: torch.Tensor, actions: torch.Tensor):
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
    feature_vector = torch.Tensor()
    for n, state in enumerate(p_states):
        n_state = torch.cat((state, actions[n]), 0)
        feature_vector = torch.cat((feature_vector, n_state), 0)
        
    return feature_vector

p_states = torch.zeros(4,400)
actions = torch.ones(4,1)

cat_prediction_feature_vector(p_states[:,::25], actions)
print()