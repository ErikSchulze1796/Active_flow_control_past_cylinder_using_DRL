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

def optimize_model(model: pt.nn.Module, x_train: pt.Tensor, y_train: pt.Tensor,
                   x_val: pt.Tensor, y_val: pt.Tensor, epochs: int=1000,
                   lr: float=0.001, save_best: str="", n_step_history: int=1) ->Tuple[List[float], List[float]]:
    """Optimize network weights based on training and validation data.

    :param model: neural network model
    :type model: pt.nn.Module
    :param x_train: features for training
    :type x_train: pt.Tensor
    :param y_train: labels for training
    :type y_train: pt.Tensor
    :param x_val: features for validation
    :type x_val: pt.Tensor
    :param y_val: labels for validation
    :type y_val: pt.Tensor
    :param epochs: number of optimization loops, defaults to 1000
    :type epochs: int, optional
    :param lr: learning rate, defaults to 0.001
    :type lr: float, optional
    :param save_best: path where to save best model; no snapshots are saved
        if empty string; defaults to ""
    :type save_best: str, optional
    :return: lists with training and validation losses for all epochs
    :rtype: Tuple[List[float], List[float]]
    """    
    criterion = pt.nn.MSELoss()
    optimizer = pt.optim.Adam(params=model.parameters(), lr=lr)
    best_val_loss, best_train_loss = 1.0e5, 1.0e5
    train_loss, val_loss = [], []
    
    torch.autograd.set_detect_anomaly(True)
    for e in range(1, epochs+1):
        optimizer.zero_grad()
        #loss = 0
        predictions_train = pt.Tensor(y_train.shape)
        # Start state prediction loop for each time step for training
        for i in range(len(x_train)-1):
            input = x_train[i].clone().detach() 
            prediction = model(input).squeeze()
            for n in range(1, n_step_history+1):
                x_train[i+n, :-1] = prediction[:-2]
            predictions_train[i, :] = prediction

        # Calculate loss
        loss = criterion(predictions_train, y_train)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
        with pt.no_grad():
            predictions_val = pt.Tensor(y_val.shape)
            # Start state prediction loop for each time step for validation
            for i in range(len(x_val)-1):
                    input = x_val[i].clone().detach()
                    prediction = model(input).squeeze()
                    x_val[i+1, :-1] = prediction[:-2]
                    predictions_val[i, :] = prediction
                    
            loss = criterion(predictions_val, y_val)
            val_loss.append(loss.item())

        print("\r", "Training/validation loss epoch {:5d}: {:10.5e}, {:10.5e}".format(e, train_loss[-1], val_loss[-1]))#, end="")
        
        if isdir(save_best):
            if train_loss[-1] < best_train_loss:
                pt.save(model.state_dict(), f"{save_best}best_model_train_{lr}.pt")
                best_train_loss = train_loss[-1]
                
            if val_loss[-1] < best_val_loss:
                pt.save(model.state_dict(), f"{save_best}best_model_val_{lr}.pt")
                best_val_loss = val_loss[-1]
                
    return train_loss, val_loss

def optimize_model_gs(model: pt.nn.Module, x_train: pt.Tensor, y_train: pt.Tensor,
                   epochs: int=1000,
                   lr: float=0.001) ->List[float]:
    """Optimize network weights based on training data.

    :param model: neural network model
    :type model: pt.nn.Module
    :param x_train: features for training
    :type x_train: pt.Tensor
    :param y_train: labels for training
    :type y_train: pt.Tensor
    :param epochs: number of optimization loops, defaults to 1000
    :type epochs: int, optional
    :param lr: learning rate, defaults to 0.001
    :type lr: float, optional
    :return: lists with training losses for all epochs
    :rtype: List[float]]
    """    
    criterion = pt.nn.MSELoss()
    optimizer = pt.optim.Adam(params=model.parameters(), lr=lr)
    train_loss = []
    
    for e in range(1, epochs+1):
        prediction = model(x_train).squeeze()

        loss = criterion(prediction, y_train)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        

        print("\r", "Training loss epoch {:5d}: {:10.5e}".format(e, train_loss[-1]), end="")
        
                
    return train_loss

def split_sequence(sequence_p_a, sequence_p_cdl, n_steps_history):
    """Based on: https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/

    Args:
        sequence (list): Sequence that should be split to get features and labels from it for prediction
        n_steps (int): Number of time steps to be included into the state vector

    Returns:
        current_t: Current state vector with possible previous states
        next_t: The label vector to be predicted
    """
    current_t, next_t = pt.Tensor(pt.Tensor()), pt.Tensor(pt.Tensor())
    for i in range(len(sequence_p_a)):
        # find the end of this pattern
        end_ix = i + n_steps_history
        # check if we are beyond the sequence
        if end_ix > len(sequence_p_a)-1:
            break

        # gather input and output parts of the pattern
        seq_current_t, seq_next_t = sequence_p_a[i:end_ix, :], sequence_p_cdl[end_ix:end_ix+1, :]
        current_t = pt.cat((current_t, seq_current_t.unsqueeze(dim=0)), 0)# 
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

def evaluate_model(model: pt.nn.Module(), features: pt.Tensor(), labels: pt.Tensor(), model_path: str()):
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
        rtype: Tuple(pt.Tensor(), pt.Tensor(), pt.Tensor())

    """
    
    model.load_state_dict((pt.load(model_path)))
    prediction = model(features).squeeze().detach()
    # we normalize the maximum error with the range of the scaled Ux,
    # which is 1-(-1)=2
    test_loss_l2 = (prediction - labels).square().mean()
    test_loss_lmax = (prediction - labels).absolute().max() #/ 2
    r2score = r2_score(labels, prediction)
    
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
        self.min = data.min(dim=0).values
        self.max = data.max(dim=0).values
        self.trained = True

    def scale(self, data):
        assert self.trained
        assert len(data.shape) == 2 # Assert dimension of input data
        data_norm = (data - self.min) / (self.max - self.min)
        return 2.0*data_norm - 1.0 # Scale between [-1, 1]

    def rescale(self, data_norm):
        assert self.trained
        assert len(data_norm.shape) == 2
        data = (data_norm + 1.0) * 0.5
        return data * (self.max - self.min) + self.min

def data_loading_all(location: str()):
    n_sensors = 400

    trajectory_files = glob(location)
    trajectory_files = sorted(trajectory_files)
    
    all_trajectory_data = np.zeros((len(trajectory_files), 399, 404))
    
    for i, file in enumerate(trajectory_files):
        print(f"Trajectory: {i}")
        coeff_data, trajectory_data, p_at_faces = read_data_from_trajectory(file, n_sensors)
        time_steps = coeff_data.t.values
        c_d = coeff_data.c_d.values
        c_l = coeff_data.c_l.values
        actions = trajectory_data.omega.values
        p_states = pt.Tensor(trajectory_data[p_at_faces].values)
        single_trajectory_data = reshape_data(time_steps, p_states, c_d, c_l, actions)
        all_trajectory_data[i] = single_trajectory_data
        
    return all_trajectory_data


def data_loading(location: str()):
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

def data_preprocessing(data, n_steps_history):
    """This function is responsible for the input data preprocessing (i.e. min-max scaling, train, val, test set splitting etc.)

    Args:
        :param data: Pytorch tensor containing the input data with the columns [t, p0-p400, cd, cl, omega]
        :type data: (pt.Tensor())
    Returns:
        :return: Lists of training, validation and test data consisting of feature and label.
        Furthermore the normalization scalers for the data are returned
        :rtype: Tuple[List(pt.Tensor(), pt.Tensor()), list(pt.Tensor(), pt.Tensor()), list(pt.Tensor(), pt.Tensor()), MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler()]
    """

    # Normalize data
    # The pressure values are normalized only by the data of the corresponding sensor
    data_norm = pt.zeros(data.shape)
    data_norm[:,0] = pt.Tensor(data[:, 0])
    scaler_pressure = MinMaxScaler()
    scaler_pressure.fit(data[:,1:-3])
    data_norm[:, 1:-3] = scaler_pressure.scale(data[:,1:-3])
    scaler_cd = MinMaxScaler()
    scaler_cd.fit(data[:,-3])
    data_norm[:,-3] = scaler_cd.scale(data[:,-3].unsqueeze(dim=0))
    scaler_cl = MinMaxScaler()
    scaler_cl.fit(data[:,-2])
    data_norm[:,-2] = scaler_cl.scale(data[:,-2].unsqueeze(dim=0))
    scaler_omega = MinMaxScaler()
    scaler_omega.fit(data[:,-1])
    data_norm[:,-1] = scaler_omega.scale(data[:,-1].unsqueeze(dim=0))
        
    # Features are all states of the time steps except for the last
    # Labels are all states of the time steps except for the first
#    feature_sequences, label_sequences = split_sequence(p_a_states, p_cdl_states, n_steps)
    feature_sequences, label_sequences = split_sequence(pt.cat((data_norm[:,1:-3], data_norm[:,-1].unsqueeze(dim=1)), dim=1), data_norm[:,1:-1], n_steps_history)

    # Test portion is relative to all data
    test_portion_rel = 0.20
    # Validation portion is relative to training set
    val_portion_rel = 0.10

    test_portion_abs = round(test_portion_rel * feature_sequences.shape[0])
    val_portion_abs = round(val_portion_rel * feature_sequences.shape[0])
    train_portion_abs = feature_sequences.shape[0] - val_portion_abs - test_portion_abs
    #print(f"Absolute size of test set: {test_portion_abs}")
    #print(f"Absolute size of validation set: {val_portion_abs}")
    #print(f"Absolute size of training set: {train_portion_abs}")
    assert (test_portion_abs + val_portion_abs + train_portion_abs) == feature_sequences.shape[0]


    # select two snapshots for testing
    probs = pt.ones(feature_sequences.shape[0])
    test_idx = pt.multinomial(probs, test_portion_abs)
    probs[test_idx] = 0.0
    val_idx = pt.multinomial(probs, val_portion_abs)
    probs[val_idx] = 0.0
    train_idx = pt.multinomial(probs, train_portion_abs)
    #print("Testing snapshots: ", test_idx)
    #print("Validation snapshots: ", val_idx)
    #print("Training snapshots: ", train_idx)

    test_data_features = feature_sequences[test_idx, :]
    test_data_labels = label_sequences[test_idx, :]
    val_data_features = feature_sequences[val_idx, :]
    val_data_labels = label_sequences[val_idx, :]
    train_data_features = feature_sequences[train_idx, :]
    train_data_labels = label_sequences[train_idx, :]
    
    # Reshape data
    
    return [train_data_features, train_data_labels], [val_data_features, val_data_labels], [test_data_features, test_data_labels], scaler_pressure, scaler_cd, scaler_cl, scaler_omega

def main():
    
    # Seeding
    pt.manual_seed(0)

    sample = 47
    location = f'../training/initial_trajectories/Data/sample_{sample}/trajectory_5/'
    n_steps_history = 1
    
    data = data_loading(location)
    train_data, val_data, test_data, scaler_pressure, scaler_cd, scaler_cl, scaler_omega = data_preprocessing(data, n_steps_history)

    ######################################
    # Training
    ######################################
    
    # Initialize the MLP
    model_params = {
        "n_inputs": n_steps_history * 401,
        "n_outputs": 402,
        "n_layers": 5,
        "n_neurons": 300,
        "activation": pt.nn.ReLU()
    }

    model = FFMLP(**model_params)
    epochs = 100
    lr = 0.0001
    save_model_in = "DRL_py_beta/model/first_training/"
    
    # Actually train model
    train_loss, val_loss = optimize_model(model, train_data[0].squeeze(), train_data[1],
                                        val_data[0], val_data[1], epochs, save_best=save_model_in, lr=lr)

    # Grid search setup
    # learning_rates = np.linspace(0.0001, 0.0001, 1)
    # n_hidden_layers = np.linspace(1, 2, 2)
    # neurons_search = np.linspace(100, 100, 1)
    
    # parameter_space = {
    #     "n_inputs": learning_rates,
    #     "n_layers": n_hidden_layers,
    #     "n_neurons": neurons_search,
    #     "lr": learning_rates
    # }
    #grid_search(feature_sequences, label_sequences, **parameter_space)    
    
    # print(evaluate_model(model, test_data[0], test_data[1], f"{save_model_in}best_model_train_0.0001.pt"))
    
    
    # Plot data
    model.load_state_dict(pt.load(f"{save_model_in}best_model_train_0.0001.pt"))
    full_pred_norm = model(pt.cat((train_data[0], val_data[0], test_data[0]), 0)).squeeze().detach()
    full_pred = pt.zeros(full_pred_norm.shape)
    full_pred[:, :-2] = scaler_pressure.rescale(full_pred_norm[:, :-2])
    full_pred[:,-2] = scaler_cd.rescale(full_pred_norm[:,-2].unsqueeze(dim=0))
    full_pred[:,-1] = scaler_cl.rescale(full_pred_norm[:,-1].unsqueeze(dim=0))

    plot_loss(train_loss, val_loss, loss_type="MSE", lr=lr, save_plots_in=save_model_in, show=True)
    # Plot drag coefficient c_D
    plot_coefficient_prediction(data[1:,0].numpy(), data[1:, -3],
                                full_pred[:, -2], y_label="c_D", save_plots_in=save_model_in, lr=lr)
    # Plot lift coefficient c_L
    plot_coefficient_prediction(data[1:,0].numpy(), data[1:, -2],
                                full_pred[:, -1], y_label="c_L", save_plots_in=save_model_in, lr=lr)
    
if __name__ == '__main__':
    main()