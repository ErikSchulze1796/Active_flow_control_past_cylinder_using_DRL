"""
This file contains the setup and trainin procedures
for the MLP pressure prediction model
"""

from typing import List, Tuple
from os.path import isdir
import datetime

import torch as pt

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
        
        # Randomize batch sample drawing
        permutation = torch.randperm(features_train.shape[0])
        n_batches = features_train.shape[0] / batch_size
        
        # Batch loop
        start = datetime.datetime.now()
        for i in range(0, features_train.shape[0], batch_size):
            # Make sure all training samples are used at least and maximal once
            indices = permutation[i:i+batch_size]
            batch_feature_train, batch_label_train = features_train[indices], labels_train[indices]

            optimizer.zero_grad()

            # Forward + backward + optimize
            prediction = model(batch_feature_train).squeeze()
            loss = criterion(prediction, batch_label_train)
            loss.backward()
            optimizer.step()
            acc_loss_train += loss.item()

        end = datetime.datetime.now()
        train_time = (end-start).total_seconds()
        # Divide loss by batch size to get average epoch loss
        train_loss.append(acc_loss_train / n_batches)

        # Validation
        start = datetime.datetime.now()
        with pt.no_grad():
            prediction = model(features_val).squeeze()
            loss = criterion(prediction, labels_val)
            val_loss.append(loss.item())
            
        end = datetime.datetime.now()
        val_time = (end-start).total_seconds()
        
        print("\r", "Training/validation loss epoch {:5d}: {:10.5e}, {:10.5e}; t_train {:2.5f}, t_val {:2.5f}".format(e, train_loss[-1], val_loss[-1], train_time, val_time), end="\r")
        
        # Save train and validation models to designated directory
        if isdir(save_best):
            if train_loss[-1] < best_train_loss:
                pt.save(model.state_dict(), f"{save_best}best_model_train_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}.pt")
                best_train_loss = train_loss[-1]
                
            if val_loss[-1] < best_val_loss:
                pt.save(model.state_dict(), f"{save_best}best_model_val_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}.pt")
                best_val_loss = val_loss[-1]
        
        if (e % 250) == 0:
            if not isdir(save_best+f"snapshot_ep{e}/"):
                os.mkdir(save_best+f"snapshot_ep{e}/")
            pt.save(model.state_dict(), f"{save_best}snapshot_ep{e}/best_model_train_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}_ep{e}.pt")
            pt.save(model.state_dict(), f"{save_best}snapshot_ep{e}/best_model_val_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}_ep{e}.pt")
    
        
        # Early stopping
        # if (train_loss[-1] <= 5e-6) and (val_loss[-1] <= 5e-6):
        #     return train_loss, val_loss

    return train_loss, val_loss

def split_sequence(data, n_steps_history: int, every_nth_element: int):
    """Splits the data into sequences of states for model input according to n_steps_history
    Based on: https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/

    Parameters
    ----------
    data : [type]
        Sequence that should be split to get features and labels from it for prediction
    n_steps_history : int
        Number of time steps to be included into the state vector
    every_nth_element : int
        Every nth pressure sensor to be kept as input

    Returns
    -------
    torch.Tensor
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

    Parameters
    ----------
    t : pt.Tensor
        time steps
    p : pt.Tensor
        pressure from sensors along surface of cylinder
    c_D : pt.Tensor
        drag coefficient for time step t
    c_L : pt.Tensor
        lift coefficient for time step t
    omega : pt.Tensor
        rotation velocity which is the taken action by the DRL agent

    Returns
    -------
    pt.Tensor
        data suitable for training; the Tensor should have the shape (N_t, 404)
        corresponding to the states of the simulation containing time, omega, drag,
        lift and the pressure values of the sensors for a number N_t of time steps
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

def evaluate_model(model: pt.nn.Module,
                    features_norm: pt.Tensor,
                    labels_norm: pt.Tensor,
                    model_path: str,
                    n_steps_history: int,
                    every_nth_element: int,
                    n_inputs: int):
    """This function evaluates the given model for given features and labels

    Parameters
    ----------
    model : pt.nn.Module
        Model to be evaluated
    features_norm : pt.Tensor
        Features the model should predict from
    labels_norm : pt.Tensor
        Labels data the prediction is tested against
    model_path : str
        Path where the model is stored
    n_steps_history : int
        Number of subsequent states to be included in input
    every_nth_element : int
        Every nth pressure sensor to be kept as input
    n_inputs : int
        Number of input features for one training example of the model

    Returns
    -------
    list, list, list, torch.Tensor, torch.Tensor, torch.Tensor
        test_loss_l2: Loss calculated using the L2 norm
        test_loss_lmax: Loss calculated using max norm
        r2score: R² (R squared) score of the prediction
        prediction_p: Predicted pressure values for all 400 sensors along the cylinder's surface
        prediction_cd: Predicted drag coefficient
        prediction_cl: Predicted lift coefficient
    """
    test_loss_l2 = []
    test_loss_lmax = []
    r2score = []
    prediction_p = pt.zeros(labels_norm[:,:-2].shape)
    prediction_cd = pt.zeros(labels_norm[:,-2].shape)
    prediction_cl = pt.zeros(labels_norm[:,-1].shape)
    model.load_state_dict((pt.load(model_path)))
    full_pred_norm = pt.zeros(labels_norm.shape)
    for idx_t, state_t in enumerate(features_norm):

        pred_norm = model(state_t).squeeze().detach()
        full_pred_norm[idx_t,:] = pred_norm
        
        prediction_p[idx_t] = pred_norm[:-2]
        prediction_cd[idx_t] = pred_norm[-2]
        prediction_cl[idx_t] = pred_norm[-1]
        # we normalize the maximum error with the range of the scaled/normalized data,
        # which is 1-(-1)=2
        test_loss_l2.append((pred_norm - labels_norm[idx_t,:]).square().mean())
        test_loss_lmax.append((pred_norm - labels_norm[idx_t,:]).absolute().max() / 2)
        r2score.append(r2_score(labels_norm[idx_t,:], pred_norm))
        
        print("\r", "L2 loss: {:1.4f}, Lmax loss: {:1.4f}, r2 score: {:1.4f}".format(test_loss_l2[idx_t], test_loss_lmax[idx_t], r2score[idx_t]), end="\r")

        if idx_t == labels_norm.shape[0]-1:
            break
        features_norm[idx_t+1,(n_steps_history-1)*n_inputs:-1] = pred_norm[:-2][::every_nth_element]
    
    return test_loss_l2, test_loss_lmax, r2score, prediction_p, prediction_cd, prediction_cl

class MinMaxScaler(object):
    """Class to scale/re-scale data to the range [-1, 1] and back by min/max scaling.
    """
    def __init__(self):
        """Constructor of MinMaxScaler class
        """
        self.min = None
        self.max = None
        self.trained = False

    def fit(self, data):
        """Stores min and max values of given data

        Parameters
        ----------
        data : torch.Tensor
            Data tensor
        """
        self.min = pt.min(data)
        self.max = pt.max(data)
        self.trained = True

    def scale(self, data):
        """Scales the data according to stored min/max values

        Parameters
        ----------
        data : torch.Tensor
            Data tensor

        Returns
        -------
        torch.Tensor
            Scaled/normalized data
        """
        assert self.trained
        assert len(data.shape) == 3 # Assert dimension of input data
        data_norm = (data - self.min) / (self.max - self.min)
        return 2.0*data_norm - 1.0 # Scale between [-1, 1]

    def rescale(self, data_norm):
        """Rescales the normalized data back

        Parameters
        ----------
        data_norm : torch.Tensor
            Normalized data tensor

        Returns
        -------
        torch.Tensor
            Denormalized data
        """
        assert self.trained
        data = (data_norm + 1.0) * 0.5
        return data * (self.max - self.min) + self.min

def data_scaling(data):
    """Conducts input data preprocessing (i.e. min-max scaling, train, val, test set splitting etc.)

    Parameters
    ----------
    data : torch.Tensor
        Pytorch tensor containing the input data with the columns [t, p0-p400, cd, cl, omega]

    Returns
    -------
    data_norm: torch.Tensor
        Normalized data
    scaler_pressure: MinMaxScaler
        Scaler object for pressure min/max scaling
    scaler_cd: MinMaxScaler
        Scaler object for cd min/max scaling
    scaler_cl: MinMaxScaler
        Scaler object for cl min/max scaling
    scaler_omega: MinMaxScaler
        Scaler object for omega min/max scaling
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

    Parameters
    ----------
    data_norm : pt.Tensor
        Normalized input data
    n_steps_history : int
        Number of subsequent states to be included in model input
    every_nth_element : int
        Every nth pressure sensor to be kept as input

    Returns
    -------
    [data_norm_features, data_norm_labels]: Dict(torch.Tensor, torch.Tensor)
        Dict of feature and label vector
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

    Parameters
    ----------
    data : pt.Tensor
        Input data to be split
    test_portion_rel : float, optional
        Test portion of the data, by default 0.20
    val_portion_rel : float, optional
        Validation portion of the data, by default 0.10

    Returns
    -------
    [train_data_features, train_data_labels], [val_data_features, val_data_labels], [test_data_features, test_data_labels]:
        Tuple(list(torch.Tensor, torch.Tensor), list(torch.Tensor, torch.Tensor), list(torch.Tensor, torch.Tensor))
        A tuple of three lists storing pairs of feature and label tensors for training, validation and test
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
    """Reduce the number of pressure sensors
        Keep Only the every nth sensor

    Parameters
    ----------
    data_norm_p : pt.Tensor
        All pressure data with dimension 400
    every_nth_element : int
        Keep only every nth sensor

    Returns
    -------
    pt.Tensor
        Reduced number of pressure sensor data
    """
    data_norm_p_reduced = pt.zeros((data_norm_p.shape[0], int(data_norm_p.shape[1]/every_nth_element)))
    for i, state in enumerate(data_norm_p):
        data_norm_p_reduced[i] = state[::every_nth_element]
    
    return data_norm_p_reduced

def main():
    
    # Seeding
    pt.manual_seed(0)

    location = "DRL_py_beta/training_pressure_model/initial_trajectories/initial_trajectory_data_pytorch_ep100_traj992_t-p400-cd-cl-omega.pt"
    data_total = pt.load(location)
    
    # Grid search set, up
    learning_rates_space = [0.0001]
    hidden_layers_space = [5]
    neurons_space = [30]
    steps_space = [2,3,4]
    n_sensors = 400
    every_nth_element = 25
    n_sensors_keep = n_sensors / every_nth_element
    assert n_sensors % every_nth_element == 0, "You can only keep an even number of sensors!"
    n_inputs = int(400 / every_nth_element + 1)
    output = 400 + 2
    batch_size = 5
    
    for lr in learning_rates_space:
        for hidden_layer in hidden_layers_space:
            for neurons in neurons_space:
                for n_steps_history in steps_space:

                    # Draw randomly k sample trajectories from all data
                    perm = torch.randperm(data_total.size(0))
                    k = 30
                    idx = perm[:k]
                    samples = data_total[idx]
                    data = samples
                    
                    data_norm, scaler_pressure, scaler_cd, scaler_cl, scaler_omega = data_scaling(data)
                    data_labeled = generate_labeled_data(data_norm, n_steps_history, every_nth_element)
                    train_data, val_data, test_data = split_data(data_labeled)
                    # data_labeled = generate_labeled_data(data, n_steps_history, every_nth_element)
                    # train_data, val_data, test_data = split_data(data_labeled)
                    
                    ######################################
                    # Training
                    ######################################
                    # DATA SHAPE: t, p400, cd, cl, omega
                    
                    p_min = pt.min(data[:, :,1:-3])
                    p_max = pt.max(data[:, :,1:-3])
                    c_d_min = pt.min(data[:,:,-3])
                    c_d_max = pt.max(data[:,:,-3])
                    c_l_min = pt.min(data[:,:,-2])
                    c_l_max = pt.max(data[:,:,-2])
                    omega_min = pt.min(data[:, :,-1])
                    omega_max = pt.max(data[:, :,-1])
                    model_params = {
                        "n_inputs": n_steps_history * n_inputs,
                        "n_outputs": output,
                        "n_layers": hidden_layer,
                        "n_neurons": neurons,
                        "activation": pt.nn.ReLU(),
                        "n_steps": n_steps_history,
                        "n_p_sensors": n_sensors_keep,
                        "p_min": p_min,
                        "p_max": p_max,
                        "c_d_min": c_d_min,
                        "c_d_max": c_d_max,
                        "c_l_min": c_l_min,
                        "c_l_max": c_l_max,
                        "omega_min": omega_min,
                        "omega_max": omega_max
                    }

                    model = FFMLP(**model_params)
                    epochs = 10000
                    save_model_in = f"DRL_py_beta/training_pressure_model/{neurons}_{hidden_layer}_{n_steps_history}_{lr}_{batch_size}_{len(data)}_p{every_nth_element}/"
                    if not isdir(save_model_in):
                        os.mkdir(save_model_in)
                    
                    # Actually train model
                    train_loss, val_loss = optimize_model(model, train_data[0], train_data[1],
                                                        val_data[0], val_data[1], n_steps_history,
                                                        n_neurons=model_params["n_neurons"],
                                                        n_layers=model_params["n_layers"],
                                                        batch_size=batch_size,
                                                        epochs=epochs, save_best=save_model_in, lr=lr)

                    # # Min/Max Scaler wrapper model
                    # wrapper = WrapperModel(model,
                    #                        p_min, p_max,
                    #                        omega_min, omega_max,
                    #                        c_d_min, c_d_max,
                    #                        c_l_min, c_l_max,
                    #                        n_steps_history,
                    #                        n_sensors_keep)

                    # train_loss, val_loss = optimize_model(wrapper, train_data[0], train_data[1],
                    #                                     val_data[0], val_data[1], n_steps_history,
                    #                                     n_neurons=model_params["n_neurons"],
                    #                                     n_layers=model_params["n_layers"],
                    #                                     batch_size=batch_size, 
                    #                                     epochs=epochs, save_best=save_model_in, lr=lr)

                    # Plot data
                    plot_loss(train_loss,
                                val_loss,
                                loss_type="MSE", 
                                lr=lr,
                                n_neurons=model_params["n_neurons"],
                                n_layers=model_params["n_layers"],
                                n_steps_history=n_steps_history,
                                save_plots_in=save_model_in, show=False)

                    model_path = "{}best_model_train_{}_n_history{}_neurons{}_layers{}.pt".format(save_model_in, lr, n_steps_history, model_params["n_neurons"], model_params["n_layers"])
                    
                    idx_test_trajectory = 1
                    time_steps = data[0,n_steps_history:,0]
                    test_features_norm = test_data[0][idx_test_trajectory,:,:]
                    test_labels_norm = test_data[1][idx_test_trajectory,:,:]
                    
                    test_loss_l2, test_loss_lmax, r2score, prediction_p, prediction_cd, prediction_cl = evaluate_model(model,
                                                                                                                       test_features_norm,
                                                                                                                       test_labels_norm,
                                                                                                                       model_path,
                                                                                                                       n_steps_history,
                                                                                                                       every_nth_element,
                                                                                                                       n_inputs)

                    labels_p_norm = test_labels_norm[:,:-2]

                    # labels[:,:-2] = scaler_pressure.rescale(labels[:,:-2])
                    labels_cd = scaler_cd.rescale(test_labels_norm[:,-2])
                    labels_cl = scaler_cl.rescale(test_labels_norm[:,-1])

                    # prediction_p = scaler_pressure.rescale(prediction_p)
                    prediction_cd = scaler_cd.rescale(prediction_cd)
                    prediction_cl = scaler_cl.rescale(prediction_cl)

                    plot_evaluation(time_steps,
                                    labels_cd,
                                    prediction_cd,
                                    labels_cl,
                                    prediction_cl,
                                    test_loss_l2,
                                    test_loss_lmax,
                                    r2score,
                                    save_model_in,
                                    lr,
                                    neurons,
                                    hidden_layer,
                                    n_steps_history)

                    plot_feature_space_error_map(time_steps = time_steps,
                                                reference = labels_p_norm,
                                                prediction = prediction_p,
                                                save_plots_in = save_model_in,
                                                lr = lr,
                                                n_neurons = model_params["n_neurons"],
                                                n_layers = model_params["n_layers"],
                                                n_steps_history = n_steps_history)
        
if __name__ == '__main__':
    main()