from os.path import isdir
from os import makedirs

import torch
from torch.utils.data.dataset import random_split

import datetime

from process_data import EnvironmentStateDataset_from_variable, MinMaxScaler, eval_multistep_model, sliding_windows_single_trajectory
from model.RNN_files.Seq2SeqLSTM import Encoder, Decoder
from model.RNN_files.train_Seq2SeqLSTM import train_LSTM
from process_data import load_trajectory_data, create_random_feature_label_data, set_n_pressure_sensors, eval_model_metrics
from model.RNN_files.plotting_LSTM import plot_loss, plot_coefficient_predictions_multiple_trajs, plot_evaluation_multiple_trajectories, plot_multiple_feature_space_error_map

if __name__ == '__main__':

    torch.manual_seed(0)
    train_mode = False
    hidden_size = 50
    layer_number = 1
    batch_size = 64
    n_epochs = 10000
    lr = 0.001
    n_p_keep = 400
    input_size = n_p_keep + 1
    seq_length_features = 4
    seq_length_labels = 3
    is_bidirectional = False
    
    location = "DRL_py_beta/training_pressure_model/initial_trajectories/data_train/initial_trajectory_data_train_set_pytorch_ep100_traj992_t-p400-cd-cl-omega.pt"
    data = load_trajectory_data(location, split=False)
    scaler = MinMaxScaler()
    randperm = torch.randperm(data.shape[0])[:100]
    data = data[randperm]

    encoder = Encoder(input_size, hidden_size, layer_number)
    decoder = Decoder(hidden_size, 402, layer_number)
    features, labels = create_random_feature_label_data(data, seq_length_features, seq_length_labels)
    if n_p_keep < 400:
        features = set_n_pressure_sensors(features, n_p_keep)

    features_norm = scaler.scale_features(features)
    labels_norm = scaler.scale_labels(labels)
    
    dataset = EnvironmentStateDataset_from_variable(features_norm, labels_norm)
    reduced_dataset_size = int(len(dataset)*0.3)
    drop_set_size = len(dataset) - reduced_dataset_size
    reduced_dataset, _ = random_split(dataset, [reduced_dataset_size, drop_set_size])
    
    train_set_size = int(len(reduced_dataset)*0.80)
    val_set_size = len(reduced_dataset) - train_set_size
    train_dataset, val_dataset = random_split(reduced_dataset, [train_set_size, val_set_size])
    save_model_folder = f"LSTM_neu{hidden_size}_lay{layer_number}_bs{batch_size}_lr{lr}_ep{n_epochs}_p{input_size-1}_bidir{is_bidirectional}_seqfeat{seq_length_features}_seqlab{seq_length_labels}/"
    save_model_path = "DRL_py_beta/training_pressure_model/LSTM/multi_step/" + save_model_folder
    if not isdir(save_model_path):
        makedirs(save_model_path)

    if train_mode:
        start = datetime.datetime.now()
        train_loss, val_loss = train_LSTM(encoder,
                                        decoder,
                                        train_dataset,
                                        val_dataset,
                                        save_model_path,
                                        batch_size,
                                        n_epochs,
                                        lr)
        end = datetime.datetime.now()
        duration = end - start
        run_data_file_path = save_model_path + f"run_data_{hidden_size}_{layer_number}_{seq_length_features}_{lr}_{batch_size}_p{n_p_keep}_eps{n_epochs}.txt"
        with open(run_data_file_path, 'a') as file:
            file.write(f'Training duration: {duration}\n')
            file.write(f'No. of epochs: {n_epochs}\n')
            file.write(f'Learning rate: {lr}\n')
            file.write(f'Batch size: {batch_size}\n')
            file.write(f'Neurons: {hidden_size}\n')
            file.write(f'Hidden layers: {layer_number}\n')
            file.write(f'No. of subsequent time steps: {seq_length_features}\n')
            file.write(f'No. of pressure sensors as input: {n_p_keep}\n')
        
    # ================================== #
    #           Evaluation               #
    
    encoder.load_state_dict(torch.load(save_model_path + f"encoder_LSTM_best_model_val_{lr}_n_history{seq_length_features}.pt"))
    decoder.load_state_dict(torch.load(save_model_path + f"decoder_LSTM_best_model_val_{lr}_n_history{seq_length_features}.pt"))
    
    location = "DRL_py_beta/training_pressure_model/initial_trajectories/data_test/initial_trajectory_data_test_set_pytorch_ep100_traj992_t-p400-cd-cl-omega.pt"
    test_data = load_trajectory_data(location, split=False)
    idxes = torch.randperm(test_data.shape[0])
    u = 5
    m = idxes[:u]
    test_data = test_data[m]
    
    all_test_l2 = []
    all_test_lmax = []
    all_r2score = []
    all_test_labels = torch.zeros(test_data.shape[0], (test_data.shape[1]-seq_length_features-(seq_length_labels-1)), 402)
    all_test_predictions = torch.zeros(test_data.shape[0], (test_data.shape[1]-seq_length_features-(seq_length_labels-1)), 402)

    for k, traj in enumerate(test_data):
        test_features, test_labels = sliding_windows_single_trajectory(traj, seq_length_features, seq_length_labels)
        test_features = torch.cat((test_features[:,:,1:-3], test_features[:,:,-1].unsqueeze(dim=2)),dim=2)
        test_labels = test_labels[:,:,1:-1]

        scaler = MinMaxScaler()
        if n_p_keep < 400:
            test_features = set_n_pressure_sensors(test_features, n_p_keep)
        test_features_norm = scaler.scale_features(test_features)
        test_labels_norm = scaler.scale_labels(test_labels)
        
        predictions = eval_multistep_model(decoder, encoder, test_features_norm, test_labels_norm)
        test_l2, test_lmax, r2score = eval_model_metrics(predictions.squeeze(), test_labels[:,0,:])
        predictions = scaler.rescale_labels(predictions)
        test_labels = scaler.rescale_labels(test_labels_norm)

        all_test_l2.append(test_l2)
        all_test_lmax.append(test_lmax)
        all_r2score.append(r2score)
        all_test_labels[k] = test_labels[:,0,:].squeeze()
        all_test_predictions[k] = predictions.squeeze()
    
    train_loss = torch.load(save_model_path + f"LSTM_train_loss_{lr}_n_history{seq_length_features}.pt")
    val_loss = torch.load(save_model_path + f"LSTM_val_loss_{lr}_n_history{seq_length_features}.pt")
    plot_loss(train_loss, val_loss, "MSE", lr, hidden_size, layer_number, seq_length_features, save_model_path)

    time_steps = test_data[:,seq_length_features:-(seq_length_labels-1),0] / 0.1
    plot_evaluation_multiple_trajectories(time_steps,
                                        all_test_l2,
                                        all_test_lmax,
                                        all_r2score,
                                        save_model_path,
                                        lr,
                                        hidden_size,
                                        layer_number,
                                        seq_length_features,
                                        u)
    
    plot_coefficient_predictions_multiple_trajs(time_steps,
                                                all_test_labels[:,:,-2],
                                                all_test_labels[:,:,-1],
                                                all_test_predictions[:,:,-2],
                                                all_test_predictions[:,:,-1],
                                                save_model_path,
                                                lr,
                                                hidden_size,
                                                layer_number,
                                                seq_length_features,
                                                rows=u)

    all_test_predictions = scaler.scale_labels(all_test_predictions)
    all_test_labels = scaler.scale_labels(all_test_labels)

    plot_multiple_feature_space_error_map(time_steps,
                                        all_test_labels[:,:,:-2],
                                        all_test_predictions[:,:,:-2],
                                        save_model_path,
                                        lr,
                                        hidden_size,
                                        layer_number,
                                        seq_length_features,
                                        rows=3,
                                        columns=2)
