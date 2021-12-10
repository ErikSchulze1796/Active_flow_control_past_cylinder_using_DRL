from datetime import time
import torch as pt

import numpy as np

from matplotlib import pyplot as plt

def plot_loss(train_loss, val_loss, loss_type: str, lr: float, n_neurons: int, n_layers: int, n_steps_history: int, save_plots_in: str, show: bool=False):
    """Plots training and validation loss

    Parameters
    ----------
    train_loss : list(float)
        List of average training loss per epoch
    val_loss : list(float)
        List of average validation loss per epoch
    loss_type : str
        Loss type (e.g. MSE, MAE, MCE)
    lr : float
        Learning rate
    n_neurons : int
        Number of neurons
    n_layers : int
        Number of layers
    n_steps_history : int
        Number of time steps being included in the feature vector
    save_plots_in : str
        Path to save the figure to
    show : bool, optional
        Flag for showing the plot, by default False
    """
    epochs = len(train_loss)
    
    fig, ax = plt.subplots()
    ax.set_title(r"Epoch vs. {} loss; Learning rate = {}; neurons={}; layers={}; steps={}".format(loss_type, lr, n_neurons, n_layers, n_steps_history))
    ax.plot(range(1, epochs+1), train_loss, lw=1.0, label="training loss")
    ax.plot(range(1, epochs+1), val_loss, lw=1.0, label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{loss_type} loss")
    ax.set_xlim(1, epochs+1)
    ax.set_yscale("log")
    fig.legend()
    if show:
        fig.show()
    fig.savefig(f"{save_plots_in}/train_val_loss_lr{lr}_neurons{n_neurons}_nlayers{n_layers}_nhistory{n_steps_history}.svg", bbox_inches="tight")

def plot_coefficient_prediction(time_steps, reference, prediction, y_label: str,
                                save_plots_in: str, lr: float, n_neurons: int, 
                                n_layers: int, n_steps_history: int):
    """Plots c_D and c_L coefficient predictions in comparison to reference

    Parameters
    ----------
    time_steps : numpy.ndarray
        Contains the time steps for each state
    reference : numpy.ndarray
        Reference c_D or c_L value
    prediction : torch.Tensor
        Model predictions
    y_label : str
        Label for the y axis
    save_plots_in : str
        Path to save the figure to
    lr : float
        Learning rate
    n_neurons : int
        Number of neurons
    n_layers : int
        Number of layers
    n_steps_history : int
        Number of time steps being included in the feature vector
    """
    assert reference.shape == prediction.shape, f"Dimensions of <reference> ({reference}) and <prediction> ({prediction}) are not matching"
    assert reference.shape == time_steps.shape, f"Dimensions of <reference> ({reference}) and <time_steps> ({time_steps}) are not matching"

    fig, ax = plt.subplots()
    ax.set_title(r"Time vs. ${}$; Learning rate = {}; neurons={}; layers={}; steps={}".format(y_label, lr, n_neurons, n_layers, n_steps_history))
    ax.plot([], label="reference", c="k")
    ax.plot([], label="prediction", c="C3", ls="--")
    ax.plot(time_steps, reference, c="k")
    ax.plot(time_steps, prediction, ls="--", c="C3")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"${}$".format(y_label))
    ax.set_xlim(0.0, max(time_steps))
    ax.set_xticks(np.arange(0.0, round(max(time_steps)+1), 1.0))

    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=[0.5, 1.12])
    fig.show()
    fig.savefig(f"{save_plots_in}/{y_label}_coefficient_pred_vs_org_lr{lr}_neurons{n_neurons}_nlayers{n_layers}_nhistory{n_steps_history}.svg", bbox_inches="tight")

def plot_feature_space_error_map(time_steps, reference, prediction, save_plots_in: str,
                                lr, n_neurons, n_layers, n_steps_history):
    
    # error = pt.transpose((prediction - reference), 0, 1)
    error = pt.transpose(((reference - prediction) / reference).abs(), 0, 1)
    
    sensors = pt.linspace(1, 400, 400)
    
    fig, ax = plt.subplots()
    pcol = plt.pcolormesh(time_steps, sensors, error.abs(), shading='auto')
    fig.colorbar(pcol, label=r"absolute error")
    ax.set_xlabel(r"t [s]")
    ax.set_ylabel(r"Sensor")
    fig.show()
    fig.savefig(f"{save_plots_in}/pressure_error_time_heatmap_lr{lr}_neurons{n_neurons}_nlayers{n_layers}_nhistory{n_steps_history}.png", bbox_inches="tight")
    