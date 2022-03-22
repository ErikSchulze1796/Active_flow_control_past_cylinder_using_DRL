"""
    This file has function to sample the trajectories and extract data,
    as states, rewards, actions, logporbs and rewards, from it.

    called in :  ppo.py
"""

from read_trajectory_data import *
from cal_R_gaes import *
from check_traj import *
from utils import cat_prediction_feature_vector, write_model_generated_trajectory_data_to_file
from model.model_env import modelEnv

# If machine = 'local' then the functions from env_local will be imported
# If machine = 'cluster' then the functions from env_cluster will be imported
machine = 'cluster'

if machine == 'local':
    from env_local import *
elif machine == 'cluster':
    from env_cluster import *
else:
    print('Provide machine')
    assert machine


def fill_buffer(env, sample, n_sensor, gamma, r_1, r_2, r_3, r_4, action_bounds):
    """
    This function is to sample trajectory and get states, actions, probabilities, rewards,
    and to calculate returns.

    Args:
        env: instance of environment class for sampling trajectories
        sample: number of iteration
        n_sensor: no of patches at the surface of cylinder
        gamma: discount factor
        r_1: coefficient for reward function
        r_2: coefficient for reward function
        r_3: coefficient for reward function
        r_4: coefficient for reward function
        action_bounds: min and max omega value

    Returns: arrays of, states, actions, rewards, returns, probabilities

    """

    # to sample the trajecties
    env.sample_trajectories(sample, action_bounds)
    
    # check the trajectory to be completed
    check_trajectories(sample)

    traj_files = glob(f'./Data/sample_{sample}' + "/*/")

    # To check if the trajectories is sampled
    n_traj = len(traj_files)
    assert n_traj > 0

    # To extract the length of trajectory
    t_traj = pd.read_csv(traj_files[0] + "trajectory.csv", sep=",", header=0)
    n_T = len(t_traj.t.values)

    # due to delayed starting behaviour the time steps set to explicit -> length of trajectory
    # choose shortest available trajectory to ensure the same length
    for i, files in enumerate(traj_files):
        # To extract the length of trajectory
        t_traj = pd.read_csv(traj_files[i] + "trajectory.csv", sep=",", header=0)
        n_T_temp = len(t_traj.t.values)
        if n_T_temp < n_T:
            n_T = n_T_temp

    # buffer initialization
    state_buffer = np.zeros((n_traj, n_T, n_sensor))
    action_buffer = np.zeros((n_traj, n_T - 1))
    reward_buffer = np.zeros((n_traj, n_T))
    return_buffer = np.zeros((n_traj, n_T))
    log_prob_buffer = np.zeros((n_traj, n_T - 1))

    for i, files in enumerate(traj_files):
        # get the dataframe from the trajectories
        coeff_data, trajectory_data, p_at_faces = read_data_from_trajectory(files, n_sensor)

        # sometimes off by 1 this fixes that
        n_T_2 = len(trajectory_data.t.values)
        if n_T_2 > n_T:
            trajectory_data = trajectory_data[:n_T]

        # state values from data frame
        states = trajectory_data[p_at_faces].values

        # action values from data frame
        actions_ = trajectory_data.omega.values
        actions = actions_[:-1]
        
        # rotation rate
        theta_ = trajectory_data.theta_sum.values
        d_theta = trajectory_data.dt_theta_sum.values

        # rewards and returns from cal_R_gaes.py -> calculate_rewards_returns
        rewards, returns = calculate_rewards_returns(r_1, r_2, r_3, r_4, coeff_data, gamma, theta_, d_theta)

        # log_probs from data frame
        log_probs_ = trajectory_data.log_p.values
        log_probs = log_probs_[:-1]

        # appending values in buffer
        state_buffer[i] = states[:n_T, :]
        action_buffer[i] = actions[:n_T-1]
        reward_buffer[i] = rewards[:n_T]
        return_buffer[i] = returns[:n_T]
        log_prob_buffer[i] = log_probs[:n_T-1]

    return state_buffer, action_buffer, reward_buffer, return_buffer, log_prob_buffer


def fill_buffer_from_environment_model_total(sample,
                                             n_sensor,
                                             gamma,
                                             r_1, r_2, r_3, r_4,
                                             trajectory_length,
                                             delta_t,
                                             n_steps,
                                             keep_nth_p,
                                             policy_model,
                                             action_bounds):
    """Replay buffer filling using the environment model

    Parameters
    ----------
    sample : int
        Episode number
    n_sensor : int
        Number of sensors used
    gamma : float
        Discount factor
    r_1 : float
        Reward function weight for cD
    r_2 : float
        Reward function weight for cL
    r_3 : float
        Reward function weight for theta
    r_4 : float
        Reward function weight for dtheta
    trajectory_length : float
        Lenght of trajectory in seconds
    delta_t : float
        Lenght of time step of CFD simulation
    n_steps : int
        Number of time steps used for prediction
    keep_nth_p : int
        Number of pressure sensors used for prediction using the environment model, 400 sensors input for agent is kept
    policy_model : torch.nn.Module
        Policy model
    action_bounds : array
        Bounds for the actuation of the cylinder rotation

    Returns
    -------
    Tuple of arrays
        Arrays sotring states, actions, rewards, returns, log_probs of trajectories generated during episode
    """
    # Load policy model
    policy_model_path = f"./env/base_case/agentRotatingWallVelocity/policy_no_torchscript.pt"
    policy_model.load_state_dict(torch.load(policy_model_path))

    # Load environmet model
    environment_model_path = f"model/trained_model/best_model_train_0.0001_n_history{n_steps}_neurons256_layers5.pt"

    # Prediction model initialization
    model_parameters = {"pmin" : -1.6963981,
                        "pmax" : 2.028614,
                        "cdmin" : 2.9635367,
                        "cdmax" : 3.4396918,
                        "clmin" : -1.8241948,
                        "clmax" : 1.7353026,
                        "omegamin" : -9.999999,
                        "omegamax" : 10.0,
                        "n_steps" : n_steps,
                        "n_sensors" : int(400 / keep_nth_p),
                        "n_inputs" : (n_steps * (int((400 / keep_nth_p)+1))),
                        "n_outputs" : 402,
                        "n_layers" : 5,
                        "n_neurons" : 256,
                        "activation" : torch.nn.ReLU()
    }
    environment_model = modelEnv(environment_model_path, **model_parameters)

    # Load single uncontrolled start state
    start_states_path = f"model/samples/uncontrolled_start_state.npy"
    start_state = torch.from_numpy(np.load(start_states_path))

    # Get number of trajectories to be used
    n_traj = 10
    assert n_traj > 0

    # Get number of time steps
    n_T = int((trajectory_length / (20 * delta_t) - 1))# / 6.5)

    # buffer initialization
    state_buffer = np.zeros((n_traj, n_T-n_steps, n_sensor))
    action_buffer = np.zeros((n_traj, n_T-n_steps - 1))
    reward_buffer = np.zeros((n_traj, n_T-n_steps))
    return_buffer = np.zeros((n_traj, n_T-n_steps))
    log_prob_buffer = np.zeros((n_traj, n_T-n_steps - 1))
    
    # Arrays for writing trajectory to file
    actions_to_file = np.zeros((n_traj, n_T-n_steps))
    action_means_to_file = np.zeros((n_traj, n_T-n_steps))
    action_log_stds_to_file = np.zeros((n_traj, n_T-n_steps))
    alphas_to_file = np.zeros((n_traj, n_T-n_steps))
    betas_to_file = np.zeros((n_traj, n_T-n_steps))
    log_probs_to_file = np.zeros((n_traj, n_T-n_steps))
    entropies_to_file = np.zeros((n_traj, n_T-n_steps))
    thetas_to_file = np.zeros((n_traj, n_T-n_steps))
    d_thetas_to_file = np.zeros((n_traj, n_T-n_steps))
    cds_to_file = np.zeros((n_traj, n_T-n_steps))
    cls_to_file = np.zeros((n_traj, n_T-n_steps))
    times_to_file = np.zeros((n_traj, n_T-n_steps))

    i = 0
    failed_counter = 0
    while i < n_traj:

        # start_states_trajectory = all_start_states_trajectory[idx].squeeze()
        start_states_trajectory = start_state.squeeze()

        # get the data from stored example tensor
        coeff_data = np.column_stack((start_states_trajectory[:,0].detach().cpu().numpy().astype(float),
                                        start_states_trajectory[:,-4].detach().cpu().numpy().astype(float),
                                        start_states_trajectory[:,-3].detach().cpu().numpy().astype(float)))
        
        # state values from stored example tensor
        # start_state shape: t, p1-p400, c_d, c_l, omega
        states = start_states_trajectory[:,1:-4].detach().cpu().numpy().astype(float)

        # action values from stored example tensor
        actions = start_states_trajectory[:,-2].detach().cpu().numpy().astype(float)
        
        # rotation rate initialized to zero since not used in reward function
        theta = []
        d_theta = []
        
        # # log_probs from stored example tensor
        log_probs = []
         
        # # omega means from stored example tensor
        action_means = start_states_trajectory[:,-1].detach().cpu().numpy().astype(float)
        
        # omega log std from stored example tensor
        action_log_stds = []
        
        alphas = []
        betas = []
        
        # At this point only the data from the first states
        # that are needed for model prediction are stored.
        ########################## Environment Prediction ##########################
        
        # Create correct time steps for trajectory, although its is not necessary for the prediction and DRL training in general
        n_time_steps = int(trajectory_length / delta_t - 1)
        start_time = coeff_data[0,0]
        time_steps = (np.arange(0,n_time_steps) * delta_t + start_time)[19::20]
        time_steps = time_steps[:n_T]

        # Environment prediction loop
        for j, time_step in enumerate(time_steps[n_steps:]):
            
            with torch.no_grad():
                # Sample action from policy network
                action, action_mean, action_log_std, alpha, beta = policy_model.select_action(np.expand_dims(states[-1,:],axis=[0,1]))
                actions = np.append(actions, np.array([action]), axis=0)
                action_means = np.append(action_means, np.array([action_mean]), axis=0)
                # action_log_stds = np.append(action_log_stds, np.array([action_log_std]), axis=0)
                action_log_stds.append(action_log_std)
                alphas.append(alpha)
                betas.append(beta)
            
            # Get last n_steps states and actions from all previous states and actions
            feature_states = states[-n_steps:,::keep_nth_p]
            feature_actions = actions[-n_steps:]
            # Concatenate feature vector using multiple states
            combined_features = cat_prediction_feature_vector(feature_states, feature_actions)
            with torch.no_grad():
                # Get state, cd and cl prediction from feature vector
                prediction = environment_model.get_prediction(combined_features).detach().cpu().unsqueeze(dim=0).numpy()
            # Append pressure values to states
            states = np.append(states, np.squeeze(prediction,axis=1)[:,:-2], axis=0)
            # Wrap up timestep, c_d, and c_l values for a single state
            coeff_state = np.array([time_step, np.squeeze(prediction,axis=1)[:,-2], np.squeeze(prediction,axis=1)[:,-1]], dtype=np.single)
            # Append everything to coefficient_data
            coeff_data = np.append(coeff_data, np.expand_dims(coeff_state, axis=0), axis=0)

            # Since r_3 and r_4 equal zero, theta and d_theta have no effect and can be set to zero for now
            theta_step = 0
            d_theta_step = 0

            theta.append(theta_step)
            d_theta.append(d_theta_step)
            

        ########################## Environment Prediction ##########################
        # Convert t, cd, cl data into dataframe for rewards function to evaluate
        coeff_data = pd.DataFrame(coeff_data, columns=["t", "c_d", "c_l"])
        # rewards and returns from cal_R_gaes.py -> calculate_rewards_returns
        rewards, returns = calculate_rewards_returns(r_1, r_2, r_3, r_4, coeff_data[n_steps:], gamma, np.array(theta), np.array(d_theta))

        # Rule out very bad rewards in order to use avoid extreme errornous trajectories
        mean_reward = np.mean(rewards)
        if mean_reward < -0.5:
            # i = i-1
            failed_counter = failed_counter + 1
            print(f"Trajectories failed: {failed_counter}", end="\r")
            continue
        
        with torch.no_grad():
            # Get log probabilities from policy network
            logpas_pred, entropy = policy_model.get_predictions(np.expand_dims(states[n_steps:],axis=0), np.expand_dims(actions[n_steps:],axis=0))

        logpas_pred = logpas_pred.squeeze()
        
        # Fill arrays with trajectory data
        times_to_file[i] = time_steps[n_steps:]
        state_buffer[i] = states[n_steps:]
        cds_to_file[i] = coeff_data.c_d.values[n_steps:]
        cls_to_file[i] = coeff_data.c_l.values[n_steps:]
        actions_to_file[i] = actions[n_steps:]
        action_means_to_file[i] = action_means[n_steps:]
        action_log_stds_to_file[i] = action_log_stds
        alphas_to_file[i] = alphas
        betas_to_file[i] = betas
        log_probs_to_file[i] = logpas_pred
        entropies_to_file[i] = entropy
        thetas_to_file[i] = theta
        d_thetas_to_file[i] = d_theta

        logpas_pred = logpas_pred[:-1]
        # appending values in buffer
        state_buffer[i] = states[n_steps:n_T, :]
        action_buffer[i] = actions[n_steps:n_T-1]
        reward_buffer[i] = rewards[:n_T-n_steps]
        return_buffer[i] = returns[:n_T-n_steps]
        log_prob_buffer[i] = logpas_pred[:n_T-n_steps-1]
        
        i = i+1

    # Write generated trajectory data to file
    for i in range(n_traj):
        write_model_generated_trajectory_data_to_file(sample,
                                                    i,
                                                    times_to_file[i],
                                                    state_buffer[i],
                                                    cds_to_file[i],
                                                    cls_to_file[i],
                                                    actions_to_file[i],
                                                    action_means_to_file[i],
                                                    action_log_stds_to_file[i],
                                                    alphas_to_file[i],
                                                    betas_to_file[i],
                                                    log_probs_to_file[i],
                                                    entropies_to_file[i],
                                                    thetas_to_file[i],
                                                    d_thetas_to_file[i])

    
    return state_buffer, action_buffer, reward_buffer, return_buffer, log_prob_buffer
