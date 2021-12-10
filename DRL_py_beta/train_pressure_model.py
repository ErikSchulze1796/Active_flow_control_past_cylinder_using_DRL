"""
This file contains the setup and trainin procedures
for the MLP pressure prediction model
"""

from glob import glob
from array import array
from typing import List, Tuple
from os.path import isdir
from scipy.stats.stats import variation

import torch as pt

import numpy as np

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from model_network_MLP import *
from read_trajectory_data import *
from plot_data import *

def optimize_model(model: pt.nn.Module, features_train: pt.Tensor, labels_train: pt.Tensor,
                   features_val: pt.Tensor, labels_val: pt.Tensor, n_steps_history: int, n_neurons: int,
                   n_layers: int, batch_size: int, epochs: int=1000,
                   lr: float=0.001, save_best: str="") ->Tuple[List[float], List[float]]:
    """Optimize network weights based on training and validation data.

    Parameters
    ----------
    model : pt.nn.Module
        Neural network to be used
    features_train : pt.Tensor
        Feature for training
    labels_train : pt.Tensor
        Labels for training
    features_val : pt.Tensor
        Features for validation
    labels_val : pt.Tensor
        Labels for validation
    n_steps_history : int
        Number of time steps being included in the feature vector
    n_neurons : int
        Number of neurons of the model
    n_layers : int
        Number of layers of the model
    batch_size : int
        Number of trajectories per batch
    epochs : int, optional
        Number of optimization loops, by default 1000
    lr : float, optional
        Learning rate, by default 0.001
    save_best : str, optional
        Path where to save best model; no snapshots are saved if empty string, by default ""

    Returns
    -------
    Tuple[List[float], List[float]]
        Lists with training and validation losses for all epochs
    """
    criterion = pt.nn.MSELoss()
    optimizer = pt.optim.Adam(params=model.parameters(), lr=lr)
    best_val_loss, best_train_loss = 1.0e5, 1.0e5
    train_loss, val_loss = [], []
    
    torch.autograd.set_detect_anomaly(True)
    for e in range(1, epochs+1):
        acc_loss_train = 0.0
        
        permutation = torch.randperm(features_train.shape[0])
        n_batches = features_train.shape[0] / batch_size
        
        for i in range(0, features_train.shape[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_feature_train, batch_label_train = features_train[indices], labels_train[indices]
            
            prediction = model(batch_feature_train)
            loss = criterion(prediction, batch_label_train)
            acc_loss_train += loss.item()
            
            loss.backward()
            optimizer.step()
        
        train_loss.append(acc_loss_train / n_batches)# / batch_size) # (features_train.shape[0] / 

        acc_loss_val = 0.0
        with pt.no_grad():
            for i in range(0,features_val.shape[0]):
                prediction = model(features_val)
                loss = criterion(prediction, labels_val)
                acc_loss_val += loss.item()
        acc_loss_val = acc_loss_val / features_val.shape[0]
        val_loss.append(loss)
        
        # acc_loss = 0.0
        # for i, traj in enumerate(features_train):
        #     optimizer.zero_grad()
        #     prediction = model(traj).squeeze()

        #     # Calculate loss
        #     loss = criterion(prediction, labels_train[i])
        #     acc_loss += loss.item()
        #     loss.backward()
        #     optimizer.step()
        
        # train_loss.append(acc_loss / features_train.shape[0])
            
        # acc_loss = 0.0
        # for i, traj in enumerate(features_val):
        #     with pt.no_grad():

        #         prediction = model(traj).squeeze()
        #         loss = criterion(prediction, labels_val[i])
        #         acc_loss += loss.item()
        # val_loss.append(acc_loss / features_val.shape[0])

        print("\r", "Training/validation loss epoch {:5d}: {:10.5e}, {:10.5e}".format(e, train_loss[-1], val_loss[-1]))#, end="")
        
        if isdir(save_best):
            if train_loss[-1] < best_train_loss:
                pt.save(model.state_dict(), f"{save_best}best_model_train_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}.pt")
                best_train_loss = train_loss[-1]
                
            if val_loss[-1] < best_val_loss:
                pt.save(model.state_dict(), f"{save_best}best_model_val_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}.pt")
                best_val_loss = val_loss[-1]
                
    return train_loss, val_loss

def split_sequence(data, n_steps_history, every_nth_element: int):
    """Based on: https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/

    Args:
        sequence (list): Sequence that should be split to get features and labels from it for prediction
        n_steps (int): Number of time steps to be included into the state vector

    Returns:
        current_t: Current state vector with possible previous states
        next_t: The label vector to be predicted
    """
    reduced_data = change_n_pressure_sensors(data[:,1:-3], every_nth_element)
    sequence_p_a = pt.cat((reduced_data, data[:,-1].unsqueeze(dim=1)), dim=1)
    # sequence_p_a = pt.cat((data[:,1:-3], data[:,-1].unsqueeze(dim=1)), dim=1)
    sequence_p_cdl = data[:,1:-1]
    
    current_t, next_t = pt.Tensor(pt.Tensor()), pt.Tensor(pt.Tensor())
    for i in range(len(sequence_p_a)):
        # Find the end of this pattern
        end_ix = i + n_steps_history
        # Check if we are beyond the sequence
        if end_ix > len(sequence_p_a)-1:
            break

        # Gather input and output parts of the pattern
        seq_current_t, seq_next_t = sequence_p_a[i:end_ix, :], sequence_p_cdl[end_ix:end_ix+1, :]
        current_t = pt.cat((current_t, seq_current_t.unsqueeze(dim=0)), 0)
        next_t = pt.cat((next_t, seq_next_t), 0)
        
    current_t = current_t.reshape((current_t.shape[0],current_t.shape[1]*current_t.shape[2]))

    return current_t, next_t

def reshape_data(t: pt.Tensor, p: pt.Tensor, c_D: pt.Tensor, c_L: pt.Tensor, omega: pt.Tensor) -> pt.Tensor:
    """Create feature and label vectors.

    :param t: time steps
    :type t: pt.Tensor
    :param p: pressure from sensors along surface of cylinder
    :type p: pt.Tensor
    :param c_D: drag coefficient for time step t
    :type c_D: pt.Tensor
    :param c_L: lift coefficient for time step t
    :type c_L: pt.Tensor()
    :param omega: rotation velocity which is the taken action by the DRL agent
    :type omega: pt.Tensor()
    :return: data suitable for training; the Tensor should have the shape (N_t, 404)
            corresponding to the states of the simulation containing time, omega, drag,
            lift and the pressure values of the sensors for a number N_t of time steps
    :rtype: pt.Tensor
    """
    assert t.shape[0] == p.shape[0]
    assert t.shape[0] == c_D.shape[0]
    assert t.shape[0] == c_L.shape[0]
    assert t.shape[0] == omega.shape[0]
    data = pt.zeros((t.shape[0], 4 + p.shape[1]))
    for i in range(t.shape[0]):
        data[i, 0] = t[i]
        data[i, 1:-3] = p[i][:]
        data[i, -3] = c_D[i]
        data[i, -2] = c_L[i]
        data[i, -1] = omega[i]
    return data

def evaluate_model(model: pt.nn.Module, feature: pt.Tensor, label: pt.Tensor, model_path: str):
    """This function evaluates the given model for given features and labels

    Args:
        :param model: Contains the model to be evaluated
        :type model: pt.nn.Module
        :param features: Contains the features the model should predict from
        :type features: pt.Tensor()
        :param labels:  Contains the labels data the prediction is tested against
        :type labels: pt.Tensor
        :param model_path: Contains the path where the model is stored
        :type model_path: str

    Returns:
        return test_loss_l2: Loss calculated using the L2 norm
        return test_loss_lmax: Loss calculated using max norm
        return r2score: R² (R squared) score of the prediction
        rtype: Tuple(pt.Tensor, pt.Tensor, pt.Tensor)

    """
    
    model.load_state_dict((pt.load(model_path)))
    prediction = model(feature).squeeze().detach()
    # we normalize the maximum error with the range of the scaled Ux,
    # which is 1-(-1)=2
    test_loss_l2 = (prediction - label).square().mean()
    test_loss_lmax = (prediction - label).absolute().max() #/ 2
    r2score = r2_score(label, prediction)
    
    print("MSE test loss: {:1.4e}".format(test_loss_l2))
    print("Lmax test loss: {:1.4e}".format(test_loss_lmax))
    print("R2 score: {:1.4e}".format(r2score))
    
    return test_loss_l2, test_loss_lmax, r2score

def grid_search(features, labels, **kwargs):
    """This function does a simple grid search for given hyperparameter optimization 
    using the given features and labels and prints the optimal parametersthem to the terminal

    Args:
        :param features: Contains the features for training
        :type features: pt.Tensor(pt.Tensor)
        :param labels: Contains the labels for training
        :type labels: pt.Tensor(pt.Tensor)
    """

    variation_neurons = kwargs.get("n_neurons")
    variation_layers = kwargs.get("n_layers")
    variation_lr = kwargs.get("lr")
    
    train_losses = np.zeros((len(variation_neurons)*len(variation_layers)*len(variation_lr), 4))
    
    idx = 0
    for n_layers in variation_layers:
        for n_neurons in variation_neurons:
            for lr in variation_lr:
                params = {
                    "n_layers": int(n_layers),
                    "n_neurons": int(n_neurons),
                }
                model = FFMLP(**params)
                train_loss = optimize_model_gs(model, features, labels, 1000, lr=lr)
                train_losses[idx, 0] = train_loss[-1]
                train_losses[idx, 1] = lr
                train_losses[idx, 2] = n_neurons
                train_losses[idx, 3] = n_layers

                print(f"\nLoss: {train_losses[idx, 0]}|Lr: {train_losses[idx, 1]}|Neurons: {int(train_losses[idx, 2])}|Layers: {int(train_losses[idx, 3])}")
                idx = idx + 1
    
    min_index = np.argmin(train_losses)
    best_params = train_losses[min_index, :]
    
    print("\nbest score: {:.3f}, best params: lr={}, n_neurons={}, n_layers={}".format(best_params[0], best_params[1], int(best_params[2]), int(best_params[3])))


class MinMaxScaler(object):
    """Class to scale/re-scale data to the range [-1, 1] and back.
    """
    def __init__(self):
        self.min = None
        self.max = None
        self.trained = False

    def fit(self, data):
        self.min = pt.min(data)
        self.max = pt.max(data)
        self.trained = True

    def scale(self, data):
        assert self.trained
        assert len(data.shape) == 3 # Assert dimension of input data
        data_norm = (data - self.min) / (self.max - self.min)
        return 2.0*data_norm - 1.0 # Scale between [-1, 1]

    def rescale(self, data_norm):
        assert self.trained
        assert len(data_norm.shape) == 2
        data = (data_norm + 1.0) * 0.5
        return data * (self.max - self.min) + self.min

def data_loading_all(location: str):
    """Loads the data located at the given location

    Parameters
    ----------
    location : str
        String storing the location of the to be loaded data

    Returns
    -------
    numpy.ndarray
        All relevant trajectory data (t, p, cd, cl, omega) located at location
    """
    n_sensors = 400

    trajectory_files = glob(location)
    trajectory_files = sorted(trajectory_files)
    
    all_trajectory_data = np.zeros((len(trajectory_files), 399, n_sensors + 4))
    
    for i, file in enumerate(trajectory_files):
        print(f"Loaded trajectories: {i}", end="\r")
        coeff_data, trajectory_data, p_at_faces = read_data_from_trajectory(file, n_sensors)
        time_steps = coeff_data.t.values
        c_d = coeff_data.c_d.values
        c_l = coeff_data.c_l.values
        actions = trajectory_data.omega.values
        p_states = pt.Tensor(trajectory_data[p_at_faces].values)
        single_trajectory_data = reshape_data(time_steps, p_states, c_d, c_l, actions)
        all_trajectory_data[i] = single_trajectory_data
        
    return all_trajectory_data


def data_loading(location: str):
    """This function loads the data that is located in location

    Args:
        :param location: String containing the file path for trajectories
        :type location: str()
    Returns:
        :return: Data loaded from location
        rtype data: pt.Tensor(pt.Tensor())
    """
    n_sensors = 400

    # Extract data from trajectory
    trajectory_file = glob(location)
    coeff_data, trajectory_data, p_at_faces = read_data_from_trajectory(trajectory_file[0], n_sensors)

    # Get drag and lift coefficients and pressure values
    time_steps = coeff_data.t.values
    c_d = coeff_data.c_d.values
    c_l = coeff_data.c_l.values
    actions = trajectory_data.omega.values
    p_states = pt.Tensor(trajectory_data[p_at_faces].values)

    # Reshape data into one pytorch tensor
    data = reshape_data(time_steps, p_states, c_d, c_l, actions)
    return data

def data_scaling(data):
    """This function is responsible for the input data preprocessing (i.e. min-max scaling, train, val, test set splitting etc.)

    Args:
        :param data: Pytorch tensor containing the input data with the columns [t, p0-p400, cd, cl, omega]
        :type data: (pt.Tensor)
    Returns:
        :return: Lists of training, validation and test data consisting of feature and label.
        Furthermore the normalization scalers for the data are returned
        :rtype: Tuple[pt.Tensor, MinMaxScaler, MinMaxScaler, MinMaxScaler, MinMaxScaler]
    """

    # Normalize data
    # The pressure values are normalized only by the data of the corresponding sensor
    data_norm = pt.Tensor(data)
    scaler_pressure = MinMaxScaler()
    scaler_pressure.fit(data_norm[:, :,1:-3])
    data_norm[:, :, 1:-3] = scaler_pressure.scale(data_norm[:, :,1:-3])
    scaler_cd = MinMaxScaler()
    scaler_cd.fit(data_norm[:, :,-3])
    data_norm[:, :,-3] = scaler_cd.scale(data_norm[:, :,-3].unsqueeze(dim=0))
    scaler_cl = MinMaxScaler()
    scaler_cl.fit(data_norm[:, :,-2])
    data_norm[:, :,-2] = scaler_cl.scale(data_norm[:, :,-2].unsqueeze(dim=0))
    scaler_omega = MinMaxScaler()
    scaler_omega.fit(data_norm[:, :,-1])
    data_norm[:, :,-1] = scaler_omega.scale(data_norm[:, :,-1].unsqueeze(dim=0))

    return data_norm, scaler_pressure, scaler_cd, scaler_cl, scaler_omega


def generate_labeled_data(data_norm: pt.Tensor, n_steps_history: int, every_nth_element: int):
    """Creates the feature and label tensors from the input data

    Args:
        :param data_norm: Normalized input data
        :type data_norm: pt.Tensor
        :param n_steps_history: Time step history of feature vectors to be included
        :type n_steps_history: int
    Returns:
        :return: Dict of feature and label vector
        :rtype: [pt.Tensor, pt.Tensor]
    """
    reduced_p_dimension = int(n_steps_history * (((data_norm.shape[2] - 4) / every_nth_element) + 1))
    
    data_norm_features = pt.zeros(data_norm.shape[0],
                                  (data_norm.shape[1]-n_steps_history),
                                  reduced_p_dimension)
    data_norm_labels = pt.zeros(data_norm.shape[0],
                                (data_norm.shape[1]-n_steps_history),
                                (data_norm.shape[2]-2))
    
    # Features are all states of the time steps except for the last
    # Labels are all states of the time steps except for the first
    for i, trajectory in enumerate(data_norm):
        data_norm_features[i,:,:], data_norm_labels[i,:,:] = split_sequence(trajectory, n_steps_history, every_nth_element)

    return [data_norm_features, data_norm_labels]


def split_data(data: pt.Tensor, test_portion_rel: float=0.20, val_portion_rel: float=0.10):
    """Splits data into train, validation and test set

    Args:
        :param data: Input data to be split
        :type data:  pt.Tensor
        :param test_portion_rel: Test portion of the data
        :type test_portion_rel: float
        :param val_portion_rel: Validation portion of the data
        :type val_portion_rel: float
    Returns:
        :return: A tuple of three lists storing pairs of feature and label tensors for training, validation and test
        :rtype: tuple(list(pt.Tensor), list(pt.Tensor), list(pt.Tensor))
    """

    test_portion_abs = round(test_portion_rel * data[0].shape[0])
    val_portion_abs = round(val_portion_rel * data[0].shape[0])
    train_portion_abs = data[0].shape[0] - val_portion_abs - test_portion_abs
    #print(f"Absolute size of test set: {test_portion_abs}")
    #print(f"Absolute size of validation set: {val_portion_abs}")
    #print(f"Absolute size of training set: {train_portion_abs}")
    assert (test_portion_abs + val_portion_abs + train_portion_abs) == data[0].shape[0]


    # select snapshots for testing
    probs = pt.ones(data[0].shape[0])
    test_idx = pt.multinomial(probs, test_portion_abs)
    probs[test_idx] = 0.0
    val_idx = pt.multinomial(probs, val_portion_abs)
    probs[val_idx] = 0.0
    train_idx = pt.multinomial(probs, train_portion_abs)
    #print("Testing snapshots: ", test_idx)
    #print("Validation snapshots: ", val_idx)
    #print("Training snapshots: ", train_idx)

    test_data_features = data[0][test_idx, :]
    test_data_labels = data[1][test_idx, :]
    val_data_features = data[0][val_idx, :]
    val_data_labels = data[1][val_idx, :]
    train_data_features = data[0][train_idx, :]
    train_data_labels = data[1][train_idx, :]
    
    return [train_data_features, train_data_labels], [val_data_features, val_data_labels], [test_data_features, test_data_labels]

def change_n_pressure_sensors(data_norm_p, every_nth_element: int):    
    data_norm_p_reduced = pt.zeros((data_norm_p.shape[0], int(data_norm_p.shape[1]/every_nth_element)))
    for i, time_step in enumerate(data_norm_p):
        data_norm_p_reduced[i] = time_step[::every_nth_element]
    
    return data_norm_p_reduced

def main():
    
    # Seeding
    pt.manual_seed(0)

    location = f'../training/initial_trajectories/Data/sample_[0-2]/trajectory_*/'
    location = f'../training/initial_trajectories/Data/sample_[6-8]/trajectory_*/'
    location = f'../training/initial_trajectories/Data/sample_*/trajectory_*/'
    # Grid search set, up
    learning_rates_space = [0.0001]#, 0.00001]
    hidden_layers_space = [4]
    neurons_space = [220]
    steps_space = [4]#, 2, 3]
    n_sensors = 400
    every_nth_element = 25
    assert n_sensors % every_nth_element == 0, "You can only keep an even number of sensors!"
    input = int(400 / every_nth_element + 1)
    output = 400 + 2
    batch_size = 8
    #n_steps_history = 1
    
    for lr in learning_rates_space:
        for hidden_layer in hidden_layers_space:
            for neurons in neurons_space:
                for n_steps_history in steps_space:
                    #neurons = neurons * n_steps_history
    
                    data = data_loading_all(location)
                    data_norm, scaler_pressure, scaler_cd, scaler_cl, scaler_omega = data_scaling(data)
                    data_labeled = generate_labeled_data(data_norm, n_steps_history, every_nth_element)
                    train_data, val_data, test_data = split_data(data_labeled)
                    ######################################
                    # Training
                    ######################################
                    
                    # Initialize the MLP
                    # model_params = {
                    #     "n_inputs": n_steps_history * 401,
                    #     "n_outputs": 402,
                    #     "n_layers": 5,
                    #     "n_neurons": 100,
                    #     "activation": pt.nn.ReLU()
                    # }
                    model_params = {
                        "n_inputs": n_steps_history * input,
                        "n_outputs": output,
                        "n_layers": hidden_layer,
                        "n_neurons": neurons,
                        "activation": pt.nn.ReLU()
                    }

                    model = FFMLP(**model_params)
                    epochs = 10000
                    # lr = 0.0001
                    # save_model_in = f"DRL_py_beta/model/first_training/220_4_3_0-0001_1_30_p10/"
                    save_model_in = f"DRL_py_beta/model/first_training/{neurons}_{hidden_layer}_{n_steps_history}_{lr}_{batch_size}_{len(data)}_p{every_nth_element}/"
                    if isdir(save_model_in):
                        pass
                    else:
                        os.mkdir(save_model_in)
                    
                    # Actually train model
                    train_loss, val_loss = optimize_model(model, train_data[0], train_data[1],
                                                        val_data[0], val_data[1], n_steps_history,
                                                        n_neurons=model_params["n_neurons"],
                                                        n_layers=model_params["n_layers"],
                                                        batch_size=batch_size,
                                                        epochs=epochs, save_best=save_model_in, lr=lr)

                    # Plot data
                    plot_loss(train_loss, val_loss, loss_type="MSE", lr=lr, n_neurons=model_params["n_neurons"], 
                                                n_layers=model_params["n_layers"], n_steps_history=n_steps_history, save_plots_in=save_model_in, show=False)

                    model.load_state_dict(pt.load("{}best_model_train_{}_n_history{}_neurons{}_layers{}.pt".format(save_model_in, lr, n_steps_history, model_params["n_neurons"], model_params["n_layers"])))
                    # full_pred_norm = model(test_data[0][0,:,:]).squeeze().detach()
                    # full_pred = pt.zeros(full_pred_norm.shape)
                    # full_pred[:, :-2] = scaler_pressure.rescale(full_pred_norm[:, :-2])
                    # full_pred[:,-2] = scaler_cd.rescale(full_pred_norm[:,-2].unsqueeze(dim=0))
                    # full_pred[:,-1] = scaler_cl.rescale(full_pred_norm[:,-1].unsqueeze(dim=0))
                    # test_data[1][0,:,-2] = scaler_cd.rescale(test_data[1][0,:,-2].unsqueeze(dim=0))
                    # test_data[1][0,:,:-2] = scaler_pressure.rescale(test_data[1][0,:,:-2])#.unsqueeze(dim=0))
                    # test_data[1][0,:,-1] = scaler_cl.rescale(test_data[1][0,:,-1].unsqueeze(dim=0))

                    test_loss_l2 = []
                    test_loss_lmax = []
                    r2score = []
                    full_pred_norm = pt.zeros(test_data[1][1,:,:].shape)
                    test_data_new = test_data[0][1,:,:]
                    for idx_t, state_t in enumerate(test_data_new[:,:]):
                        # if idx_t >= int(test_data[0][0,:,:].shape[0])-1:
                        #     break
                        pred_norm = model(state_t).squeeze().detach()
                        full_pred_norm[idx_t,:] = pred_norm
                        test_loss_l2.append((pred_norm - test_data[1][1,idx_t,:]).square().mean())
                        test_loss_lmax.append((pred_norm - test_data[1][1,idx_t,:]).absolute().max()) #/ 2
                        r2score.append(r2_score(test_data[1][1,idx_t,:], pred_norm))
                        print(f"L2 loss: {test_loss_l2[idx_t]}")
                        print(f"Lmax loss: {test_loss_lmax[idx_t]}")
                        print(f"r2 score: {r2score[idx_t]}")
                        if idx_t == test_data_new[:,:].shape[0]-1:
                            break
                        test_data_new[idx_t+1,(n_steps_history-1)*input:-1] = pred_norm[:-2][::every_nth_element]

                    test_data[1][1,:,-2] = scaler_cd.rescale(test_data[1][1,:,-2].unsqueeze(dim=0))
                    test_data[1][1,:,:-2] = scaler_pressure.rescale(test_data[1][1,:,:-2])#.unsqueeze(dim=0))
                    test_data[1][1,:,-1] = scaler_cl.rescale(test_data[1][1,:,-1].unsqueeze(dim=0))

                    full_pred = pt.zeros(full_pred_norm.shape)
                    full_pred[:, :-2] = scaler_pressure.rescale(full_pred_norm[:, :-2])
                    full_pred[:,-2] = scaler_cd.rescale(full_pred_norm[:,-2].unsqueeze(dim=0))
                    full_pred[:,-1] = scaler_cl.rescale(full_pred_norm[:,-1].unsqueeze(dim=0))

                    fig, ax = plt.subplots(5,1)
                    fig.suptitle(f"FFNN, lr={lr}, neurons={neurons}, layers={hidden_layer}, steps={n_steps_history}")
                    ax[0].plot(range(0,len(test_loss_l2)), test_loss_l2)
                    ax[1].plot(range(0,len(test_loss_lmax)), test_loss_lmax)
                    ax[2].plot(range(0,len(r2score)), r2score)
                    ax[3].plot(data[0,n_steps_history:,0],full_pred[:,-2], label="prediction", c="C3", ls="--")
                    ax[3].plot(data[0,n_steps_history:,0],test_data[1][1,:,-2], label="reference", c="k")
                    ax[4].plot(data[0,n_steps_history:,0],full_pred[:,-1], label="prediction", c="C3", ls="--")
                    ax[4].plot(data[0,n_steps_history:,0],test_data[1][1,:,-1], label="reference", c="k")
                    
                    ax[0].set_xlabel("Time step")
                    ax[1].set_xlabel("Time step")
                    ax[2].set_xlabel("Time step")
                    ax[3].set_xlabel("Time [s]")
                    ax[4].set_xlabel("Time [s]")
                    
                    ax[0].set_ylabel("L2 loss")
                    ax[1].set_ylabel("Lmax loss")
                    ax[2].set_ylabel("R² Score")
                    ax[3].set_ylabel(r"$c_D$")
                    ax[4].set_ylabel(r"$c_L$")
                    
                    # ax[0].set_title("L2 loss vs time steps")
                    # ax[1].set_title("Lmax loss vs time steps")
                    # ax[2].set_title("R² score vs time steps")
                    # ax[3].set_title(r"$c_D$ vs time")
                    # ax[4].set_title(r"$c_L$ vs time")
                    
                    fig.legend(loc="upper right")
                    # fig.tight_layout(pad=0.4, w_pad=0.01, h_pad=1.0)
                    fig.show()
                    fig.savefig(f"{save_model_in}/evalutation_lr{lr}_neurons{neurons}_nlayers{hidden_layer}_nhistory{n_steps_history}.svg")#, bbox_inches="tight")
                    
                    # error = pt.transpose((((test_data[1][0,:,:-2] - full_pred[:,:-2]))), 0, 1)
                    # absolute_error = error.abs()
                    # sensors = pt.linspace(1, 400, 400)
                    
                    # fig, ax = plt.subplots(2,1)
                    # pcol1 = ax[0].pcolormesh(data[0,1:,0], sensors, error, shading='auto')
                    # pcol2 = ax[1].pcolormesh(data[0,1:,0], sensors, absolute_error, shading='auto')
                    # fig.colorbar(pcol1, ax=ax[0], label=r"Error")
                    # fig.colorbar(pcol2, ax=ax[1], label=r"Absolut error")
                    # ax[0].set_xlabel(r"t [s]")
                    # ax[1].set_xlabel(r"t [s]")
                    # ax[0].set_ylabel(r"Sensor")
                    # ax[1].set_ylabel(r"Sensor")
                    # fig.show()
                    
                    
                    
                    # path = "{}best_model_train_{}_n_history{}_neurons{}_layers{}.pt".format(save_model_in, lr, n_steps_history, model_params["n_neurons"], model_params["n_layers"])
                    # evaluate_model(model, train_data[0][11,:,:] ,train_data[1][11,:,:],path)
                    
                    # # Plot drag coefficient c_D
                    # plot_coefficient_prediction(data[0,n_steps_history:,0], test_data[1][0,:,-2], full_pred[:,-2], #data[0,n_steps_history:,-3]
                    #                             y_label="c_D", save_plots_in=save_model_in, 
                    #                             lr=lr, n_neurons=model_params["n_neurons"], 
                    #                             n_layers=model_params["n_layers"], n_steps_history=n_steps_history)
                    # # # Plot lift coefficient c_L
                    # plot_coefficient_prediction(data[0,n_steps_history:,0], test_data[1][0,:,-1], full_pred[:,-1], 
                    #                             y_label="c_L", save_plots_in=save_model_in, 
                    #                             lr=lr, n_neurons=model_params["n_neurons"], 
                    #                             n_layers=model_params["n_layers"], n_steps_history=n_steps_history)
                    # Plot feature space heat map
                    
                    plot_feature_space_error_map(pt.Tensor(data[0,n_steps_history:,0]).detach(), pt.Tensor(test_data[1][1,:,:-2]).detach(),
                                                full_pred[:,:-2], save_model_in, lr=lr, n_neurons=model_params["n_neurons"], 
                                                n_layers=model_params["n_layers"], n_steps_history=n_steps_history)
        
if __name__ == '__main__':
    main()