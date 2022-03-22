"""
This file contains the setup and trainin procedures
for the feed forward neural network used as pressure prediction environment model
"""

import datetime
from os import makedirs
from os.path import isdir

import torch as pt
from torch.utils.data.dataset import random_split

from environment_dataset import EnvironmentStateDataset
from model.model_training_weigthed_features import optimize_model_weighted_features
from model.nnEnvironmentModel import FFNN, WrapperModel
from plotting.plot_data import plot_loss


def main():
    
    # ========================
    # Seeding
    pt.manual_seed(0)
    # Threading
    # pt.set_num_threads(16)

    # ========================
    # Model parameters
    hidden_layer = 5
    neurons = 256
    n_steps_history = 30
    n_sensors = 400
    # Keep only every nth pressure sensor of input state, uniformly distributed over cylinder surface
    # Only for environment model input not for DRL agent input
    every_nth_element = 25 # corresponds to 16 pressure sensors
    n_sensors_keep = int(n_sensors / every_nth_element)
    assert n_sensors % every_nth_element == 0, "You can only keep an even number of sensors!"
    # n_inputs = no. of p sensors / every_nth_element + omega value
    n_inputs = int(400 / every_nth_element + 1)
    output = 400 + 2

    model_params = {
        "n_inputs": n_steps_history * n_inputs,
        "n_outputs": output,
        "n_layers": hidden_layer,
        "n_neurons": neurons,
        "activation": pt.nn.ReLU(),
        "n_steps": n_steps_history,
        "n_p_sensors": n_sensors_keep,
    }

    # ========================
    # Train parameters
    batch_size = 10000
    epochs = 10000
    lr = 0.0001
    do_retrain = False

    # ========================
    # Parameters for data scaling
    p_min = -1.6963981
    p_max = 2.028614
    c_d_min = 2.9635367
    c_d_max = 3.4396918
    c_l_min = -1.8241948
    c_l_max = 1.7353026
    omega_min = -9.999999
    omega_max = 10.0

    # ========================
    # Data loading
    location_train_features = f"./training_pressure_model/initial_trajectories/initial_trajectory_data_features_train_set_pytorch_steps30_ep40_p16-omega.pt"
    location_train_labels = f"./training_pressure_model/initial_trajectories/initial_trajectory_data_labels_train_set_pytorch_steps30_ep40_p400-cd-cl.pt"

    dataset = EnvironmentStateDataset(location_train_features, location_train_labels)

    train_set_size = int(len(dataset)*0.90)
    val_set_size = len(dataset) - train_set_size
    train_dataset, val_dataset = random_split(dataset, [train_set_size, val_set_size])

    train_loader = pt.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = pt.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # =======================
    # Model instantiation
    model = FFNN(**model_params)
    # Min/Max Scaler wrapper model
    wrapper = WrapperModel(model,
                            p_min, p_max,
                            omega_min, omega_max,
                            c_d_min, c_d_max,
                            c_l_min, c_l_max,
                            n_steps_history,
                            n_sensors_keep)
    # Model saving
    save_model_in = f"./training_pressure_model/FFNN/weighted/{neurons}_{hidden_layer}_{n_steps_history}_{lr}_{batch_size}_p{n_sensors_keep}_eps{epochs}/"
    if not isdir(save_model_in):
        makedirs(save_model_in)
    # Model loading if retraining
    if do_retrain:
        load_model_from = f"./training_pressure_model/FFNN/best_model_train_0.0001_n_history30_neurons50_layers5_backup.pt"
        wrapper.load_state_dict(pt.load(load_model_from))

    # =======================
    # Train loop
    start = datetime.datetime.now()
    train_loss, val_loss = optimize_model_weighted_features(wrapper, train_loader, val_loader,
                                            batch_size,
                                            n_steps_history,
                                            n_neurons=model_params["n_neurons"],
                                            n_layers=model_params["n_layers"],
                                            epochs=epochs, save_best=save_model_in, lr=lr)
    end = datetime.datetime.now()
    duration = end - start
    
    # =======================
    # Write training meta data to file
    run_data_file_path = save_model_in + f"run_data_{neurons}_{hidden_layer}_{n_steps_history}_{lr}_{batch_size}_p{n_sensors_keep}_eps{epochs}.txt"
    with open(run_data_file_path, 'a') as file:
        file.write(f'Training duration: {duration}\n')
        file.write(f'No. of epochs: {epochs}\n')
        file.write(f'Learning rate: {lr}\n')
        file.write(f'Batch size: {batch_size}\n')
        file.write(f'No. of Samples: {len(train_dataset)}\n')
        file.write(f'Neurons: {neurons}\n')
        file.write(f'Hidden layers: {hidden_layer}\n')
        file.write(f'No. of subsequent time steps: {n_steps_history}\n')
        file.write(f'No. of pressure sensors as input: {n_sensors_keep}\n')

    # =======================
    # Plotting
    plot_loss(train_loss,
                val_loss,
                loss_type="MSE", 
                lr=lr,
                n_neurons=model_params["n_neurons"],
                n_layers=model_params["n_layers"],
                n_steps_history=n_steps_history,
                save_plots_in=save_model_in, show=False)

if __name__ == '__main__':
    main()
