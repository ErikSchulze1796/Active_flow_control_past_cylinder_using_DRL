"""
    This file has function to sample the trajectories and extract data,
    as states, rewards, actions, logporbs and rewards, from it.

    called in :  ppo.py
"""

from read_trajectory_data import *
from cal_R_gaes import *
from check_traj import *
from utils import cat_prediction_feature_vector

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

# def fill_buffer_from_environment_model(sample, n_sensor, gamma, r_1, r_2, r_3, r_4, action_bounds, trajectory_length, delta_t, n_steps, keep_nth_p, policy_model):
    
#     # Sample the trajectories to get the first few time steps to predict from
#     # env.sample_trajectories(sample, action_bounds)
    
#     # # check the trajectory to be completed
#     # env.check_trajectories(sample)

#     traj_files = glob(f'./model/samples/sample_{sample}' + "/*/")

#     # To check if the trajectories is sampled
#     n_traj = len(traj_files)
#     assert n_traj > 0
    
#     # # To extract the length of trajectory
#     # t_traj = pd.read_csv(traj_files[0] + "trajectory.csv", sep=",", header=0)
#     # n_T = len(t_traj.t.values)

#     # # due to delayed starting behaviour the time steps set to explicit -> length of trajectory
#     # # choose shortest available trajectory to ensure the same length
#     # for i, files in enumerate(traj_files):
#     #     # To extract the length of trajectory
#     #     t_traj = pd.read_csv(traj_files[i] + "trajectory.csv", sep=",", header=0)
#     #     n_T_temp = len(t_traj.t.values)
#     #     if n_T_temp < n_T:
#     #         n_T = n_T_temp

#     n_T = trajectory_length / delta_t

#     # buffer initialization
#     state_buffer = np.zeros((n_traj, n_T, n_sensor))
#     action_buffer = np.zeros((n_traj, n_T - 1))
#     reward_buffer = np.zeros((n_traj, n_T))
#     return_buffer = np.zeros((n_traj, n_T))
#     log_prob_buffer = np.zeros((n_traj, n_T - 1))

#     # Prediction model initialization
#     from model.model_env import modelEnv
#     model_location = "path/to/model"
#     model_parameters = {"pmin" : -1.6963981,
#                         "pmax" : 2.028614,
#                         "cdmin" : 2.9635367,
#                         "cdmax" : 3.4396918,
#                         "clmin" : -1.8241948,
#                         "clmax" : 1.7353026,
#                         "omegamin" : -9.999999,
#                         "omegamax" : 10.0,
#                         "n_steps" : 4,
#                         "n_sensors" : 16
#     }
#     model = modelEnv(model_location, **model_parameters)

#     for i, files in enumerate(traj_files):
#         # get the dataframe from the trajectories
#         coeff_data, trajectory_data, p_at_faces = read_data_from_trajectory(files, n_sensor)

#         # sometimes off by 1 this fixes that
#         n_T_2 = len(trajectory_data.t.values)
#         if n_T_2 > n_T:
#             trajectory_data = trajectory_data[:n_T]

#         # state values from data frame reduced by keeping only every nth p value
#         states = trajectory_data[p_at_faces].values

#         # action values from data frame
#         actions_ = trajectory_data.omega.values
        
#         # rotation rate
#         theta_ = trajectory_data.theta_sum.values
#         d_theta = trajectory_data.dt_theta_sum.values
        
#        # log_probs from data frame
#         log_probs_ = trajectory_data.log_p.values
         
#         # At this point the buffers are only filled with the data from the 
#         # first states that are needed for model prediction
#         ########################## Environment Prediction ##########################
        
#         # Environment prediction loop
#         for time_step in range(trajectory_data.t.values[-1], (trajectory_length+delta_t), delta_t):

#             # Get log probabilities from policy network
#             logpas_pred, entropies_pred = policy_model.get_predictions(states[-1], actions_[-1])
#             log_probs_.append(logpas_pred)
            
#             # Sample action from policy network
#             action = policy_model.select_action(states[-1])
#             actions_.append(action)
            
#             # Get last n_steps states and actions from all previous states and actions
#             feature_states = states[-n_steps:][::keep_nth_p]
#             feature_actions = actions_[-n_steps:]
#             # Concatenate feature vector using multiple states
#             features = cat_prediction_feature_vector(feature_states, feature_actions)
            
#             # Get state, cd and cl prediction from feature vector
#             prediction = model.get_prediction(features)
            
#             states.append(prediction[:-2])
#             coeff_state = [time_step, prediction[-2], prediction[-1]]
#             coeff_data.append(coeff_state)

#             # Since r_3 and r_4 equal zero, theta and d_theta have no effect and can be set to zero for now
#             theta_step = 0
#             d_theta_step = 0
#             theta_.append(theta_step)
#             d_theta.append(d_theta_step)
            
#         ########################## Environment Prediction ##########################
        
#         # rewards and returns from cal_R_gaes.py -> calculate_rewards_returns
#         rewards, returns = calculate_rewards_returns(r_1, r_2, r_3, r_4, coeff_data, gamma, theta_, d_theta)

#         actions = actions_[:-1]

#         log_probs = log_probs_[:-1]

#         # appending values in buffer
#         state_buffer[i] = states[:n_T, :]
#         action_buffer[i] = actions[:n_T-1]
#         reward_buffer[i] = rewards[:n_T]
#         return_buffer[i] = returns[:n_T]
#         log_prob_buffer[i] = log_probs[:n_T-1]
    
#     return state_buffer, action_buffer, reward_buffer, return_buffer, log_prob_buffer

def fill_buffer_from_environment_model_total(sample, n_sensor, gamma, r_1, r_2, r_3, r_4, trajectory_length, delta_t, n_steps, keep_nth_p, policy_model):
    
    model_path = "./model/trained_model/best_model_train_0.0001_n_history4_neurons50_layers4_ep4500.pt"
    start_states_path = "./model/trained_model/1000_random_start_states_p400_steps4.pt"
    start_states_log_p_path = "./model/samples/1000_random_start_states_p400_steps4_log_probs.pt"
    
    start_states_data = torch.load(start_states_path)
    start_states_log_p = torch.load(start_states_log_p_path)


    # To check if the trajectories is sampled
    n_traj = start_states_data.shape[0]
    assert n_traj > 0
    
    # # To extract the length of trajectory
    # t_traj = pd.read_csv(traj_files[0] + "trajectory.csv", sep=",", header=0)
    # n_T = len(t_traj.t.values)

    # # due to delayed starting behaviour the time steps set to explicit -> length of trajectory
    # # choose shortest available trajectory to ensure the same length
    # for i, files in enumerate(traj_files):
    #     # To extract the length of trajectory
    #     t_traj = pd.read_csv(traj_files[i] + "trajectory.csv", sep=",", header=0)
    #     n_T_temp = len(t_traj.t.values)
    #     if n_T_temp < n_T:
    #         n_T = n_T_temp

    n_T = trajectory_length / delta_t

    # buffer initialization
    state_buffer = np.zeros((n_traj, n_T, n_sensor))
    action_buffer = np.zeros((n_traj, n_T - 1))
    reward_buffer = np.zeros((n_traj, n_T))
    return_buffer = np.zeros((n_traj, n_T))
    log_prob_buffer = np.zeros((n_traj, n_T - 1))

    # Prediction model initialization
    from model.model_env import modelEnv
    model_parameters = {"pmin" : -1.6963981,
                        "pmax" : 2.028614,
                        "cdmin" : 2.9635367,
                        "cdmax" : 3.4396918,
                        "clmin" : -1.8241948,
                        "clmax" : 1.7353026,
                        "omegamin" : -9.999999,
                        "omegamax" : 10.0,
                        "n_steps" : 4,
                        "n_sensors" : 16
    }
    model = modelEnv(model_path, **model_parameters)

    for i, start_state in enumerate(start_states_data):
        # get the data from stored example tensor
        coeff_data = np.ndarray(start_state[:,0].detach().cpu().numpy(),
                                start_state[:,-3].detach().cpu().numpy(),
                                start_state[:,-2].detach().cpu().numpy())
        
        # state values from stored example tensor
        states = start_state[:,1:-3].detach().cpu().numpy()

        # action values from stored example tensor
        actions_ = start_state[:,-1].detach().cpu().numpy()
        
        # rotation rate initialized to zero since not used in reward function
        theta_ = np.zeros(n_steps)
        d_theta = np.zeros(n_steps)
        
       # log_probs from stored example tensor
        log_probs_ = start_states_log_p[i,:].detach().cpu().numpy()
         
        # At this point the buffers are only filled with the data from the 
        # first states that are needed for model prediction
        ########################## Environment Prediction ##########################
        
        # Environment prediction loop
        for time_step in range(coeff_data[0,-1], (trajectory_length+delta_t), delta_t):
            
            with torch.no_grad():
            # Get log probabilities from policy network
                logpas_pred, entropies_pred = policy_model.get_predictions(states[-1], actions_[-1])
                log_probs_.append(logpas_pred)
                
                # Sample action from policy network
                action = policy_model.select_action(states[-1])
                actions_.append(action)
            
            # Get last n_steps states and actions from all previous states and actions
            feature_states = states[-n_steps:,::keep_nth_p]
            feature_actions = actions_[-n_steps:]
            # Concatenate feature vector using multiple states
            combined_features = cat_prediction_feature_vector(feature_states, feature_actions)
            
            with torch.no_grad():
                # Get state, cd and cl prediction from feature vector
                prediction = model.get_prediction(combined_features).detach().squeeze()
                
            states.append(prediction[:-2])
            coeff_state = [time_step, prediction[-2], prediction[-1]]
            coeff_data.append(coeff_state)

            # Since r_3 and r_4 equal zero, theta and d_theta have no effect and can be set to zero for now
            theta_step = 0
            d_theta_step = 0
            theta_.append(theta_step)
            d_theta.append(d_theta_step)
            
        ########################## Environment Prediction ##########################
        
        # rewards and returns from cal_R_gaes.py -> calculate_rewards_returns
        rewards, returns = calculate_rewards_returns(r_1, r_2, r_3, r_4, coeff_data, gamma, theta_, d_theta)

        actions = actions_[:-1]

        log_probs = log_probs_[:-1]

        # appending values in buffer
        state_buffer[i] = states[:n_T, :]
        action_buffer[i] = actions[:n_T-1]
        reward_buffer[i] = rewards[:n_T]
        return_buffer[i] = returns[:n_T]
        log_prob_buffer[i] = log_probs[:n_T-1]
    
    return state_buffer, action_buffer, reward_buffer, return_buffer, log_prob_buffer