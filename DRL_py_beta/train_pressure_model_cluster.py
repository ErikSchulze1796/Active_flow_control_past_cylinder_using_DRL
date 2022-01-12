"""
This file contains the setup and trainin procedures
for the MLP pressure prediction model
"""

from os.path import isdir
from os import makedirs
import datetime

import torch as pt

from model.nnEnvironmentModel import FFNN
from model.model_training import optimize_model
from model.preprocessing.preprocessing import data_scaling, generate_labeled_data, split_data
from model.model_evaluation import evaluate_model
from plotting.plot_data import plot_loss, plot_evaluation, plot_feature_space_error_map

def main():
    
    # Seeding
    pt.manual_seed(0)

    location = "training_pressure_model/initial_trajectories/initial_trajectory_data_pytorch_ep100_traj992_t-p400-cd-cl-omega.pt"
    data = pt.load(location)
    
    # Grid search set, up
    epochs = 10000
    learning_rates_space = [0.0001]
    hidden_layers_space = [5]
    neurons_space = [50]
    steps_space = [4]
    n_sensors = 400
    # Keep only every nth pressure sensor of input state
    every_nth_element = 25
    n_sensors_keep = int(n_sensors / every_nth_element)
    assert n_sensors % every_nth_element == 0, "You can only keep an even number of sensors!"
    n_inputs = int(400 / every_nth_element + 1)
    output = 400 + 2
    batch_size = 32
    for lr in learning_rates_space:
        for hidden_layer in hidden_layers_space:
            for neurons in neurons_space:
                for n_steps_history in steps_space:

                    # Draw randomly k sample trajectories from all data
                    # perm = pt.randperm(data.size(0))
                    # k = 30
                    # idx = perm[:k]
                    # samples = data[idx]
                    # data = samples
                    
                    data_labeled = generate_labeled_data(data, n_steps_history, every_nth_element)
                    train_data_unscaled, val_data_unscaled, test_data_unscaled = split_data(data_labeled)
                    train_data, val_data, test_data, scaler_pressure, scaler_cd, scaler_cl, scaler_omega = data_scaling(train_data_unscaled, val_data_unscaled, test_data_unscaled)
                    # data_labeled = generate_labeled_data(data, n_steps_history, every_nth_element)
                    # train_data, val_data, test_data = split_data(data_labeled)
                    
                    ######################################
                    # Training
                    ######################################
                    # DATA SHAPE: t, p400, cd, cl, omega
                    
                    # p_min = pt.min(data[:, :,1:-3])
                    # p_max = pt.max(data[:, :,1:-3])
                    # c_d_min = pt.min(data[:,:,-3])
                    # c_d_max = pt.max(data[:,:,-3])
                    # c_l_min = pt.min(data[:,:,-2])
                    # c_l_max = pt.max(data[:,:,-2])
                    # omega_min = pt.min(data[:, :,-1])
                    # omega_max = pt.max(data[:, :,-1])
                    model_params = {
                        "n_inputs": n_steps_history * n_inputs,
                        "n_outputs": output,
                        "n_layers": hidden_layer,
                        "n_neurons": neurons,
                        "activation": pt.nn.ReLU(),
                        "n_steps": n_steps_history,
                        "n_p_sensors": n_sensors_keep,
                    }
                    #     "p_min": p_min,
                    #     "p_max": p_max,
                    #     "c_d_min": c_d_min,
                    #     "c_d_max": c_d_max,
                    #     "c_l_min": c_l_min,
                    #     "c_l_max": c_l_max,
                    #     "omega_min": omega_min,
                    #     "omega_max": omega_max
                    # }

                    model = FFNN(**model_params)
                    save_model_in = f"training_pressure_model/thesis_quality/{neurons}_{hidden_layer}_{n_steps_history}_{lr}_{batch_size}_{len(data)}_p{n_sensors_keep}_eps{epochs}/"
                    if not isdir(save_model_in):
                        makedirs(save_model_in)
                    
                    # Actually train model
                    start = datetime.datetime.now()
                    train_loss, val_loss = optimize_model(model, train_data[0], train_data[1],
                                                        val_data[0], val_data[1], n_steps_history,
                                                        n_neurons=model_params["n_neurons"],
                                                        n_layers=model_params["n_layers"],
                                                        batch_size=batch_size,
                                                        epochs=epochs, save_best=save_model_in, lr=lr)
                    end = datetime.datetime.now()
                    duration = end - start
                    run_data_file_path = save_model_in + f"run_data_{neurons}_{hidden_layer}_{n_steps_history}_{lr}_{batch_size}_{len(data)}_p{n_sensors_keep}_eps{epochs}.txt"
                    with open(run_data_file_path, 'a') as file:
                        file.write(f'Training duration: {duration}\n')
                        file.write(f'No. of epochs: {epochs}\n')
                        file.write(f'Learning rate: {lr}\n')
                        file.write(f'Batch size: {batch_size}\n')
                        file.write(f'Neurons: {neurons}\n')
                        file.write(f'Hidden layers: {hidden_layer}\n')
                        file.write(f'No. of subsequent time steps: {n_steps_history}\n')
                        file.write(f'No. of trajectories: {len(data)}\n')
                        file.write(f'No. of pressure sensors as input: {n_sensors_keep}\n')

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
                    
                    #idx_test_trajectory = 1
                    perm = pt.randperm(test_data[0].size(0))
                    k = 1
                    idx_test_trajectory = perm[:k]
                    time_steps = data[0,n_steps_history:,0]
                    test_features_norm = test_data[0][idx_test_trajectory,:,:].squeeze()
                    test_labels_norm = test_data[1][idx_test_trajectory,:,:].squeeze()
                    
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