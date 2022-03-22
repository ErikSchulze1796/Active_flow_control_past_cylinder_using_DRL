from os import makedirs
from os.path import isdir
from unicodedata import bidirectional
from venv import create
from sklearn.metrics import mean_gamma_deviance

import torch
from torch import nn
from torch.utils.data.dataset import random_split
from torch.autograd import Variable

import numpy as np

from matplotlib import pyplot as plt

from model.RNN_files.LSTM import LSTMEnvironment

def train_network(model, train_data, val_data, save_best, batch_size=64, n_epochs=1, learning_rate=0.001):

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, drop_last=False)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    best_val_loss, best_train_loss = 1.0e5, 1.0e5
    train_loss, val_loss = [], []

    for epoch in range(n_epochs):
        model.train()
        acc_loss_train = 0.0
        acc_loss_val = 0.0

        for k, (feature_batch, label_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            
            prediction = model(feature_batch)
            loss = criterion(prediction, label_batch)
            loss.backward()
            optimizer.step()
            acc_loss_train += loss.item() * len(feature_batch)
        train_loss.append(acc_loss_train/len(train_loader))

        with torch.no_grad():
            for j, (feature_batch_val, label_batch_val) in enumerate(val_loader):
                prediction = model(feature_batch_val).squeeze()
                loss = criterion(prediction, label_batch_val.squeeze())
                acc_loss_val += loss.item() * len(feature_batch_val)
            val_loss.append(acc_loss_val/len(val_loader))

        print("Training/validation loss epoch {:5d}: {:10.5e}, {:10.5e} ".format(epoch, train_loss[-1], val_loss[-1]))

        n_steps_history = next(iter(train_loader))[0].shape[1]
        if isdir(save_best):
            if train_loss[-1] < best_train_loss:
                pt.save(model.state_dict(), f"{save_best}LSTM_best_model_train_{learning_rate}_n_history{n_steps_history}.pt")
                best_train_loss = train_loss[-1]
                
            if val_loss[-1] < best_val_loss:
                pt.save(model.state_dict(), f"{save_best}LSTM_best_model_val_{learning_rate}_n_history{n_steps_history}.pt")
                best_val_loss = val_loss[-1]
        
        pt.save(train_loss, f"{save_best}LSTM_train_loss_{learning_rate}_n_history{n_steps_history}.pt")
        pt.save(val_loss, f"{save_best}LSTM_val_loss_{learning_rate}_n_history{n_steps_history}.pt")

    return train_loss, val_loss

def sliding_windows(data, seq_length):
    x = torch.Tensor(data.shape[0]-seq_length,seq_length,17)
    y = torch.Tensor(data.shape[0]-seq_length,402)

    for i in range(len(data)-seq_length):
        _x = torch.cat((data[i:(i+seq_length), 1:-3:25], data[i:(i+seq_length), -1].unsqueeze(dim=1)), dim=1)
        _y = data[i+seq_length, 1:-1]
        x[i] = _x
        y[i] = _y

    return torch.Tensor(x),torch.Tensor(y)

import torch as pt
from torch.utils.data import Dataset

class EnvironmentStateDataset_not_from_file(Dataset):
    def __init__(self, features, labels):
        self.state_features = features
        self.state_labels = labels

    def __len__(self):
        return len(self.state_labels)

    def __getitem__(self, idx):
        state = self.state_features[idx]
        label = self.state_labels[idx]
        
        return state, label
    
def plot_multiple_feature_space_error_map(time_steps,
                                        reference,
                                        predictions,
                                        save_plots_in: str,
                                        lr,
                                        n_neurons,
                                        n_layers,
                                        n_steps_history,
                                        rows: int,
                                        columns: int):
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import pandas as pd
    
    fig, axes = plt.subplots(rows, columns)
    # faxes = axes.ravel()
    fig.suptitle(f"FFNN Test: Relative Pressure Error per Sensor vs. Time \n LR={lr}, h-Neurons={n_neurons}, h-Layers={n_layers}, Steps={n_steps_history}")
    plt.rcParams.update({'font.family':'serif'})
    # Load sensor positions in polar coordinates
    sensor_positions = pd.read_csv("DRL_py_beta/plotting/centers.csv", sep=" ", header=0)

    p_max_error = 0
    error_rel = ((reference - predictions) / 2).abs() * 100
    max_error = pt.max((error_rel))
    if max_error > p_max_error:
        p_max_error = max_error

    error_rel = ((reference - predictions) / 2).abs() * 100
    # Sort sensor errors by position using the polar angle
    for i, row in enumerate(error_rel):
        error_rel[i,:] = pt.Tensor([x for _, x in sorted(zip(pt.from_numpy(sensor_positions.polar.values), row), key=lambda pair: pair[0])])
    
    error_rel = pt.transpose(error_rel, 0, 1)
    sensors = pt.linspace(1, 400, 400)
    
    pcol = axes.pcolormesh(time_steps, sensors, error_rel, shading='auto', vmin=0, vmax=p_max_error)

    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    fig.colorbar(pcol, cax=cax, label="Relative Prediction Error [%]")
    
    axes.set_xlabel(r"$t^{*}$")
    axes.set_ylabel(r"Sensor")
    # axes.axvline(x=train_size, c='r', linestyle='--', label="Val start")
    # axes.axvline(x=(train_size+val_size), c='r', linestyle=':', label="Test start")

    
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=0.5)

    fig.savefig(f"{save_plots_in}thesis_LSTM_singlestep/relative_pressure_error_time_heatmap_lr{lr}_neurons{n_neurons}_nlayers{n_layers}_nhistory{n_steps_history}.png", bbox_inches="tight")

def create_labeled_data(data, steps):
    features = torch.Tensor(data.shape[0]*(data.shape[1]-steps), steps, 17)
    labels = torch.Tensor(data.shape[0]*(data.shape[1]-steps), 402)
    
    for i, traj in enumerate(data):
        
        features_traj, labels_traj = sliding_windows(traj, steps)
        features[(i*len(features_traj)):((i+1)*len(features_traj)),:,:] = features_traj
        labels[(i*len(labels_traj)):((i+1)*len(labels_traj)),:] = labels_traj
    
    return features, labels
        

def plot_evaluation_multiple_trajectories(time_steps,
                                          l2,
                                          lmax,
                                          r2,
                                          labels_cd,
                                          labels_cl,
                                          predictions_cd,
                                          predictions_cl,
                                          save_plots_in: str,
                                          lr: float,
                                          neurons: int,
                                          hidden_layers: int,
                                          n_steps_history: int,
                                          rows: int,
                                          columns: int):
        
    fig, ax = plt.subplots(rows, columns)
    font = {'fontname': 'serif'}
    plt.rcParams.update({'font.family':'serif'})
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig.suptitle(f"FFNN Test Performance \n LR={lr}, h-Neurons={neurons}, h-Layers={hidden_layers}, Steps={n_steps_history}", **font, y=1.02)

    for idx, time_steps_traj in enumerate(time_steps):
        
        ax[idx][0].plot(time_steps_traj, l2[idx], linewidth=1.0, c="b", label=r"$L_2$ loss")
        ax[idx][1].plot(time_steps_traj, lmax[idx], linewidth=1.0, c="g", label=r"$L_{max}$ loss")
        ax[idx][2].plot(time_steps_traj, r2[idx], linewidth=1.0, c="r", label="R² score")

        ax[idx][0].set_ylabel(r"$L_2$ loss", **font)
        ax[idx][1].set_ylabel(r"$L_{max}$ loss", **font)
        ax[idx][2].set_ylabel("R² score", **font)

        ax[idx][0].get_yaxis().set_label_coords(-0.54,0.5)
        ax[idx][1].get_yaxis().set_label_coords(-0.4,0.5)
        ax[idx][2].get_yaxis().set_label_coords(-0.54,0.5)

        # ax[0].legend(prop={'size': 9, 'family':'serif'})#, loc="right")
        # ax[1].legend(prop={'size': 9, 'family':'serif'})#, loc="right")
        # ax[2].legend(prop={'size': 9, 'family':'serif'})#, loc="right")

        ax[idx][0].grid()
        ax[idx][1].grid()
        ax[idx][2].grid()
        ax[idx][0].set_xlabel(r"$t^{*}$", **font)
        ax[idx][1].set_xlabel(r"$t^{*}$", **font)
        ax[idx][2].set_xlabel(r"$t^{*}$", **font)

    for axis1 in ax:
        for axis2 in axis1:
            for tick in axis2.get_xticklabels():
                tick.set_fontname("serif")
            for tick in axis2.get_yticklabels():
                tick.set_fontname("serif")

    ax_handles = []
    ax_labels = []
    for axis2 in ax[0]:
        handle, label = axis2.get_legend_handles_labels()
        ax_handles.extend(handle)
        ax_labels.extend(label)

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.5)

    fig.legend(ax_handles, ax_labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=3)
    fig.savefig(f"{save_plots_in}/evalutation_lr{lr}_neurons{neurons}_nlayers{hidden_layers}_nhistory{n_steps_history}.svg", bbox_inches="tight")
    
    # Plotting for cd and cl predicitons

    fig, ax = plt.subplots(rows,columns-1)
    fig.suptitle(f"FFNN Test Prediction \n LR={lr}, Neurons={neurons}, Hidden Layers={hidden_layers}, Steps={n_steps_history}", **font, y=1.06)

    for idx, time_steps_traj in enumerate(time_steps):

        ax[idx][0].plot(time_steps_traj, predictions_cd[idx], label=r"pred. drag $c_D$", c="C3", ls="--", linewidth=1.0)
        ax[idx][0].plot(time_steps_traj, labels_cd[idx], label=r"real drag $c_D$", c="k", linewidth=1.0)
        ax[idx][1].plot(time_steps_traj, predictions_cl[idx], label=r"pred. lift $c_L$", c="C1", ls="--", linewidth=1.0)
        ax[idx][1].plot(time_steps_traj, labels_cl[idx], label=r"real lift $c_L$", c="b", linewidth=1.0)

        ax[idx][0].set_ylabel(r"$c_D$", **font)
        ax[idx][1].set_ylabel(r"$c_L$", **font)

        ax[idx][0].get_yaxis().set_label_coords(-0.25,0.5)
        ax[idx][1].get_yaxis().set_label_coords(-0.2,0.5)

        ax[idx][0].grid()
        ax[idx][1].grid()

    ax[-1][0].set_xlabel(r"$t^{*}$", **font)
    ax[-1][1].set_xlabel(r"$t^{*}$", **font)

    for axis1 in ax:
        for axis2 in axis1:
            for tick in axis2.get_xticklabels():
                tick.set_fontname("serif")
            for tick in axis2.get_yticklabels():
                tick.set_fontname("serif")

    ax_handles = []
    ax_labels = []
    for axis2 in ax[0]:
        handle, label = axis2.get_legend_handles_labels()
        ax_handles.extend(handle)
        ax_labels.extend(label)

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.28, hspace=0.5)

    fig.legend(ax_handles, ax_labels, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=2)
    fig.savefig(f"{save_plots_in}/cdclprediction_lr{lr}_neurons{neurons}_nlayers{hidden_layers}_nhistory{n_steps_history}.svg", bbox_inches="tight")

def plot_cd_cl_loss(time_steps, dataY_plot, data_predict, train_loss, val_loss, save_model_path, n_epochs, lr, hidden_size, layer_number, steps):
    plt.rcParams.update({'font.family':'serif'})
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["grid.linestyle"]="dashed"

    fig, axes = plt.subplots(2,1)
    axes[0].plot(time_steps, dataY_plot[:,-2], label=r"$c_D$ real")
    axes[0].plot(time_steps, data_predict[:,-2], label=r"$c_D$ pred.")
    axes[0].set_ylabel(r"$c_D$")
    axes[0].set_xlabel(r"$t^{*}$")
    axes[1].plot(time_steps, dataY_plot[:,-1], label=r"$c_L$ real")
    axes[1].plot(time_steps, data_predict[:,-1], label=r"$c_L$ pred.")
    axes[1].set_ylabel(r"$c_L$")
    axes[1].set_xlabel(r"$t^{*}$")
    for ax in axes:
        ax.grid()

    for ax in axes:
        ax.legend(prop={'size': 8, 'family':'serif'})
    path = f"{save_model_path}"
    if not isdir:
        makedirs(path)
    fig.tight_layout()
    path = f"{save_model_path}thesis_LSTM_singlestep/"
    if not isdir(path):
        makedirs(path)
    plt.savefig(f"{save_model_path}thesis_LSTM_singlestep/LSTM_cdlcl_ep{n_epochs}_lr{lr}_steps{steps}_layers{layer_number}_hiddensize{hidden_size}.svg", bbox_inches="tight")

    fig, axes = plt.subplots(1,1)
    axes.plot(train_loss, label=r"Train loss")
    axes.plot(val_loss, label=r"Val Loss")
    axes.set_yscale("log")
    axes.set_ylabel(r"MSE loss")
    axes.set_xlabel("# Epochs")
    axes.grid()

    axes.legend(prop={'size': 8, 'family':'serif'})
    path = f"{save_model_path}"
    if not isdir:
        makedirs(path)
    fig.tight_layout()
    path = f"{save_model_path}thesis_LSTM_singlestep/"
    if not isdir(path):
        makedirs(path)
    plt.savefig(f"{save_model_path}thesis_LSTM_singlestep/LSTM_loss_ep{n_epochs}_lr{lr}_steps{steps}_layers{layer_number}_hiddensize{hidden_size}.svg", bbox_inches="tight")

import random
import os
def set_seed(seed = 1234):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def run():

    set_seed(0)
    # ====== STATICS ======
    batch_size = 64
    input_size = 17       # number of pressure sensors + action
    hidden_size = 128     # number of hidden neurons
    layer_number = 5       # number of layers
    output_size = 402      # Next State containing all pressure sensors + c_d + c_l
    is_bidirectional = False # Bidirectionality of the LSTM
    # =====================
    steps = 30
    n_train_samples = 1000
    lr = 0.001
    n_epochs = 100
    multi_step = 2
    # =====================

    # location_train_data = "DRL_py_beta/training_pressure_model/initial_trajectories/full_initial_trajectory_data_pytorch_ep100_traj992_t-p400-cd-cl-omega.pt"
    location_train_data = "DRL_py_beta/training_pressure_model/initial_trajectories/data_train/initial_trajectory_data_train_set_pytorch_ep100_traj992_t-p400-cd-cl-omega.pt"


    # full_train_data = torch.load(location_train_data)

    p_min = -1.6963981
    p_max = 2.028614
    c_d_min = 2.9635367
    c_d_max = 3.4396918
    c_l_min = -1.8241948
    c_l_max = 1.7353026
    omega_min = -9.999999
    omega_max = 10.0


    from preprocessing.preprocessing import MinMaxScaler
    scaler_pressure = MinMaxScaler()
    scaler_pressure.fit(pt.Tensor([p_min, p_max]))
    scaler_cd = MinMaxScaler()
    scaler_cd.fit(pt.Tensor([c_d_min, c_d_max]))
    scaler_cl = MinMaxScaler()
    scaler_cl.fit(pt.Tensor([c_l_min, c_l_max]))
    scaler_omega = MinMaxScaler()
    scaler_omega.fit(pt.Tensor([omega_min, omega_max]))

    # full_train_data[:,:,1:-3] = scaler_pressure.scale(full_train_data[:,:,1:-3])
    # full_train_data[:,:,-3] = scaler_cd.scale(full_train_data[:,:,-3])
    # full_train_data[:,:,-2] = scaler_cl.scale(full_train_data[:,:,-2])
    # full_train_data[:,:,-1] = scaler_omega.scale(full_train_data[:,:,-1])

    # features, labels = create_labeled_data(full_train_data, steps)

    # idxes = torch.randperm(features.shape[0])
    # features = features[idxes]
    # labels = labels[idxes]


    # features = features[:n_train_samples]
    # labels = labels[:n_train_samples]

    # train_size = int(len(features) * 0.70)
    # val_size = len(features) - train_size

    # data_features = Variable(features)
    # data_labels = Variable(labels)

    # data_train_features = Variable(features[:train_size])
    # data_train_labels = Variable(labels[:train_size])

    # data_val_features = Variable(features[train_size:])
    # data_val_labels = Variable(labels[train_size:])

    # data_test_features = Variable(features[(train_size+val_size):len(features)])
    # data_test_labels = Variable(labels[(train_size+val_size):len(labels)])

    # dataset = EnvironmentStateDataset(location_train_features, location_train_labels)
    # reduced_dataset_size = int(len(dataset)*0.05)
    # drop_set_size = len(dataset) - reduced_dataset_size
    # reduced_dataset, _ = random_split(dataset, [reduced_dataset_size, drop_set_size])

    # train_set_size = int(len(reduced_dataset)*0.80)
    # val_set_size = len(reduced_dataset) - train_set_size
    # train_dataset, val_dataset = random_split(reduced_dataset, [train_set_size, val_set_size])

    save_model_folder = f"LSTM_neu{hidden_size}_lay{layer_number}_bs{batch_size}_lr{lr}_ep{n_epochs}_p{input_size-1}_bidir{is_bidirectional}_steps{steps}/"
    save_model_path = "DRL_py_beta/training_pressure_model/LSTM/" + save_model_folder
    if not isdir(save_model_path):
        makedirs(save_model_path)
    
    environment_model = LSTMEnvironment(input_size, hidden_size, layer_number, output_size, is_bidirectional, multi_step)

    # dataset_train = EnvironmentStateDataset_not_from_file(data_train_features, data_train_labels)
    # dataset_val = EnvironmentStateDataset_not_from_file(data_val_features, data_val_labels)

    # train_loss, val_loss = train_network(environment_model, dataset_train, dataset_val, save_model_path, batch_size, n_epochs, lr)

    environment_model.load_state_dict(torch.load(save_model_path + f"LSTM_best_model_train_{lr}_n_history{steps}.pt"))
    environment_model.eval()

    location_test_trajs = "DRL_py_beta/training_pressure_model/initial_trajectories/data_test/initial_trajectory_data_test_set_pytorch_ep100_traj992_t-p400-cd-cl-omega.pt"

    test_trajs = torch.load(location_test_trajs)
    idxes = torch.randperm(test_trajs.shape[0])
    test_traj = test_trajs[idxes[0]]
    time_steps = test_traj[steps:,0] / 0.1

    test_features, test_labels = sliding_windows(test_traj, steps)

    test_features[:,:,:-1] = scaler_pressure.scale(test_features[:,:,:-1])
    test_features[:,:,-1] = scaler_omega.scale(test_features[:,:,-1])

    test_labels[:,:-2] = scaler_pressure.scale(test_labels[:,:-2])
    test_labels[:,-2] = scaler_cd.scale(test_labels[:,-2])
    test_labels[:,-1] = scaler_cl.scale(test_labels[:,-1])

    pred_features = torch.zeros(test_features.shape)
    pred_features[0,:,:] = test_features[0,:,:]
    pred_features[:,:,-1] = test_features[:,:,-1]
    predictions = torch.zeros(test_labels.shape)
    
    for i, step in enumerate(pred_features):
        train_predict = environment_model.get_prediction(step.unsqueeze(dim=0)).detach()
        predictions[i] = train_predict
        if i == len(pred_features)-1:
            break
        pred_features[i+1,:-1,:-1] = pred_features[i,1:,:-1]
        pred_features[i+1,-1,:-1] = train_predict[:,:-2:25]

    data_predict = predictions
    dataY_plot = test_labels

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    data_predict[:,:-2] = scaler_pressure.rescale(data_predict[:,:-2])
    data_predict[:,-2] = scaler_cd.rescale(data_predict[:,-2])
    data_predict[:,-1] = scaler_cl.rescale(data_predict[:,-1])

    dataY_plot[:,:-2] = scaler_pressure.rescale(dataY_plot[:,:-2])
    dataY_plot[:,-2] = scaler_cd.rescale(dataY_plot[:,-2])
    dataY_plot[:,-1] = scaler_cl.rescale(dataY_plot[:,-1])

    train_loss = torch.load(save_model_path + f"LSTM_train_loss_{lr}_n_history{steps}.pt")
    val_loss = torch.load(save_model_path + f"LSTM_val_loss_{lr}_n_history{steps}.pt")
        
    mean_cd_labels = dataY_plot[:,-2].mean()
    mean_cl_labels = dataY_plot[:,-1].mean()
    mean_cd_predictions = data_predict[:,-2].mean()
    mean_cl_predictions = data_predict[:,-1].mean()
    max_cd_labels = dataY_plot[:,-2].max()
    max_cl_labels = dataY_plot[:,-1].max()
    max_cd_predictions = data_predict[:,-2].max()
    max_cl_predictions = data_predict[:,-1].max()
    mae_cd = mean_absolute_error(dataY_plot[:,-2], data_predict[:,-2])
    mae_cl = mean_absolute_error(dataY_plot[:,-1], data_predict[:,-1])
    mae_p = mean_absolute_error(dataY_plot[:,:-2], data_predict[:,:-2])
    mse_cd = mean_squared_error(dataY_plot[:,-2], data_predict[:,-2])
    mse_cl = mean_squared_error(dataY_plot[:,-1], data_predict[:,-1])
    mse_p = mean_squared_error(dataY_plot[:,:-2], data_predict[:,:-2])


    data_predict[:,:-2] = scaler_pressure.scale(data_predict[:,:-2])
    dataY_plot[:,:-2] = scaler_pressure.scale(dataY_plot[:,:-2])

    plot_cd_cl_loss(time_steps,
                    dataY_plot,
                    data_predict,
                    train_loss,
                    val_loss,
                    save_model_path,
                    n_epochs,
                    lr,
                    hidden_size,
                    layer_number,
                    steps)

    plot_multiple_feature_space_error_map(time_steps,
                                        dataY_plot[:,:-2],
                                        data_predict[:,:-2],
                                        save_model_path, lr, hidden_size, layer_number, steps, 1, 1)
    
    
    max_p_error = torch.max(((dataY_plot[:,:-2] - data_predict[:,:-2]) / 2).abs())
    min_p_error = torch.min(((dataY_plot[:,:-2] - data_predict[:,:-2]) / 2).abs())
    
    mean_min_max_data = {"mae_cd": mae_cd,
                         "mae_cl": mae_cl,
                         "mae_p": mae_p,
                         "mse_cd": mse_cd,
                         "mse_cl": mse_cl,
                         "mse_p": mse_p,
                         "mean_cd_labels": mean_cd_labels,
                         "mean_cl_labels": mean_cl_labels,
                         "mean_cd_predictions": mean_cd_predictions,
                         "mean_cl_predictions": mean_cl_predictions,
                         "max_cd_labels": max_cd_labels,
                         "max_cl_labels": max_cl_labels,
                         "max_cd_predictions": max_cd_predictions,
                         "max_cl_predictions": max_cl_predictions,
                         "max_p_error": max_p_error,
                         }

    np.save(save_model_path + "thesis_LSTM_singlestep/mae_mse_mean_min_max_data_cdclp.npy", mean_min_max_data)

    import json

    with open((save_model_path + 'thesis_LSTM_singlestep/mae_mse_mean_min_max_data_cdclp.txt'), 'w') as file:
        file.write(json.dumps(str(mean_min_max_data))) # use `json.loads` to do the reverse

    
# Entry point
run()