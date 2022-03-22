from typing import Tuple, List
from os.path import isdir
from os import makedirs
import datetime

import torch as pt

def weighted_criterion(y_prediction: pt.Tensor, y_label: pt.Tensor):
    assert y_label.shape == y_prediction.shape, "Dimension of weigth matrix, prediction and label tensor must match!"
    
    weight_matrix = pt.ones((y_prediction.shape))
    weight_matrix[:,:-2] = weight_matrix[:,:-2] * (2/402)
    weight_matrix[:,-2:] = weight_matrix[:,-2:] * (400/402)
    
    weighted_mse = ((y_prediction - y_label).square() * weight_matrix).mean()
    
    return weighted_mse

def optimize_model_weighted_features(model: pt.nn.Module, train_loader, val_loader,
                   batch_size: int, n_steps_history: int, n_neurons: int,
                   n_layers: int, epochs: int=1000,
                   lr: float=0.001, save_best: str="", prefix="") ->Tuple[List[float], List[float]]:
    """Optimize network weights based on training and validation data.

    Parameters
    ----------
    model : pt.nn.Module
        Neural network to be used
    train_loader : torch.Dataloader
        Training data containing features and labels
    val_loader : torch.Dataloader
        validation data containing features and labels
    batch_size : int
        Batch size
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
    prefix : str
        Prefix for saving the model

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
        acc_loss_val = 0.0
                
        # Batch loop
        start = datetime.datetime.now()
        for i, (feature_batch, label_batch) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            # Forward + backward + optimize
            prediction = model(feature_batch).squeeze()
            loss = weighted_criterion(prediction, label_batch)
            loss.backward()
            optimizer.step()
            acc_loss_train += loss.item() * len(feature_batch)
        train_loss.append(acc_loss_train/len(train_loader))

        end = datetime.datetime.now()
        train_time = (end-start).total_seconds()
        # Divide loss by batch size to get average epoch loss

        # Validation
        start = datetime.datetime.now()
        with pt.no_grad():
            for j, (feature_batch_val, label_batch_val) in enumerate(val_loader):
                prediction = model(feature_batch_val).squeeze()
                loss = weighted_criterion(prediction, label_batch_val)
                acc_loss_val += loss.item() * len(feature_batch_val)
            val_loss.append(acc_loss_val/len(val_loader))
                
        end = datetime.datetime.now()
        val_time = (end-start).total_seconds()
        
        if train_loss[-1] < best_train_loss:
            train_tracker = "decreasing"
        else:
            train_tracker = "increasing"
        
        if val_loss[-1] < best_val_loss:
            val_tracker = "decreasing"
        else:
            val_tracker = "increasing"
        
        print("\r", "Training/validation loss epoch {:5d}: {:10.5e} ({}), {:10.5e} ({}); t_train {:2.5f}, t_val {:2.5f}".format(e, train_loss[-1], train_tracker, val_loss[-1], val_tracker, train_time, val_time), end="\r")
        
        # Save train and validation models to designated directory
        if isdir(save_best):
            if train_loss[-1] < best_train_loss:
                pt.save(model.state_dict(), f"{save_best}{prefix}weigted_best_model_train_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}.pt")
                best_train_loss = train_loss[-1]
                
            if val_loss[-1] < best_val_loss:
                pt.save(model.state_dict(), f"{save_best}{prefix}weigted_best_model_val_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}.pt")
                best_val_loss = val_loss[-1]
        
        pt.save(train_loss, f"{save_best}{prefix}train_loss_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}.pt")
        pt.save(val_loss, f"{save_best}{prefix}val_loss_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}.pt")
        
        if (e % 250) == 0:
            if not isdir(save_best+f"snapshot_ep{e}/"):
                makedirs(save_best+f"snapshot_ep{e}/")
            pt.save(model.state_dict(), f"{save_best}snapshot_ep{e}/best_model_train_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}_ep{e}.pt")
            pt.save(model.state_dict(), f"{save_best}snapshot_ep{e}/best_model_val_{lr}_n_history{n_steps_history}_neurons{n_neurons}_layers{n_layers}_ep{e}.pt")
    
        
        # Early stopping
        # if (train_loss[-1] <= 5e-6) and (val_loss[-1] <= 5e-6):
        #     return train_loss, val_loss

    return train_loss, val_loss
