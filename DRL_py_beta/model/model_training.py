from typing import Tuple, List
from os.path import isdir
from os import makedirs
import datetime

import torch as pt

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
    
    pt.autograd.set_detect_anomaly(True)
    for e in range(1, epochs+1):
        acc_loss_train = 0.0
        
        # Randomize batch sample drawing
        permutation = pt.randperm(features_train.shape[0])
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
                makedirs(save_best+f"snapshot_ep{e}/")
            pt.save(model.state_dict(), f"{save_best}snapshot_ep{e}/best_model_train_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}_ep{e}.pt")
            pt.save(model.state_dict(), f"{save_best}snapshot_ep{e}/best_model_val_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}_ep{e}.pt")
    
        
        # Early stopping
        # if (train_loss[-1] <= 5e-6) and (val_loss[-1] <= 5e-6):
        #     return train_loss, val_loss

    return train_loss, val_loss
