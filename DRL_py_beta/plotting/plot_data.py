import torch as pt

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

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
    font = {'fontname': 'serif'}
    plt.rcParams.update({'font.family':'serif'})
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    ax.set_title((f"Epoch vs. {loss_type} loss\n" + fr"$\eta_{{lr}}$ = {lr}; h-Neurons={n_neurons}; h-Layers={n_layers}; Steps={n_steps_history}"), **font)

    ax.plot(range(1, epochs+1), train_loss, lw=1.0, label="Training loss")
    ax.plot(range(1, epochs+1), val_loss, lw=1.0, label="Validation loss")

    ax.set_xlabel("Epoch", **font)
    ax.set_ylabel(f"{loss_type} loss", **font)
    ax.set_xlim(1, epochs+1)
    ax.set_yscale("log")

    for tick in ax.get_xticklabels():
        tick.set_fontname("serif")
    for tick in ax.get_yticklabels():
        tick.set_fontname("serif")

    ax.legend()
    if show:
        fig.show()
    fig.tight_layout()
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
    assert reference.shape == prediction.shape, f"Dimensions of <reference> ({reference.shape}) and <prediction> ({prediction.shape}) are not matching"
    assert reference.shape == time_steps.shape, f"Dimensions of <reference> ({reference.shape}) and <time_steps> ({time_steps.shape}) are not matching"

    fig, ax = plt.subplots()
    ax.set_title(r"Time vs. ${}$; LR = {}; h-Neurons={}; h-Layers={}; Steps={}".format(y_label, lr, n_neurons, n_layers, n_steps_history))
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
    
    error_rel = pt.transpose(((reference - prediction) / 2).abs(), 0, 1)
    
    sensors = pt.linspace(1, 400, 400)
    
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.family':'serif'})

    fig.suptitle(f"FFNN Test: Relative Pressure Error per Sensor vs. Time \n LR={lr}, h-Neurons={n_neurons}, h-Layers={n_layers}, Steps={n_steps_history}")
    pcol = plt.pcolormesh(time_steps, sensors, error_rel, shading='auto')
    fig.colorbar(pcol, label=r"Relative Error")
    
    ax.set_xlabel(r"t [s]")
    ax.set_ylabel(r"Sensor")
    fig.savefig(f"{save_plots_in}/relative_pressure_error_time_heatmap_lr{lr}_neurons{n_neurons}_nlayers{n_layers}_nhistory{n_steps_history}.png", bbox_inches="tight")

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

    
    fig, axes = plt.subplots(rows, columns)
    faxes = axes.ravel()
    fig.suptitle(f"FFNN Test: Relative Pressure Error per Sensor vs. Time \n LR={lr}, h-Neurons={n_neurons}, h-Layers={n_layers}, Steps={n_steps_history}")
    plt.rcParams.update({'font.family':'serif'})
    # Load sensor positions in polar coordinates
    sensor_positions = pd.read_csv("plotting/centers.csv", sep=" ", header=0)

    p_max_error = 0
    for idx, time_steps_traj in enumerate(time_steps):
        error_rel = ((reference[idx] - predictions[idx]) / 2).abs() * 100
        max_error = pt.max((error_rel))
        if max_error > p_max_error:
            p_max_error = max_error

    for idx, time_steps_traj in enumerate(time_steps):
        error_rel = ((reference[idx] - predictions[idx]) / 2).abs() * 100
        # Sort sensor errors by position using the polar angle
        for i, row in enumerate(error_rel):
            error_rel[i,:] = pt.Tensor([x for _, x in sorted(zip(pt.from_numpy(sensor_positions.polar.values), row), key=lambda pair: pair[0])])
        
        error_rel = pt.transpose(error_rel, 0, 1)
        sensors = pt.linspace(1, 400, 400)
        
        pcol = faxes[idx].pcolormesh(time_steps_traj, sensors, error_rel, shading='auto', vmin=0, vmax=p_max_error)

        divider = make_axes_locatable(faxes[idx])
        cax = divider.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(pcol, cax=cax, label="Rel. Error [%]")
        
        faxes[idx].set_xlabel(r"$t^{*}$")
        faxes[idx].set_ylabel(r"Sensor")
    
    if (len(predictions) % 2) != 0:
        fig.delaxes(faxes[-1])

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=0.5)

    fig.savefig(f"{save_plots_in}/relative_pressure_error_time_heatmap_lr{lr}_neurons{n_neurons}_nlayers{n_layers}_nhistory{n_steps_history}.png", bbox_inches="tight")


def plot_evaluation(time_steps, labels_cd, prediction_cd, labels_cl, prediction_cl, test_loss_l2, test_loss_lmax, r2score, save_plots_in: str, lr: float, neurons: int, hidden_layers: int, n_steps_history: int):

    fig, ax = plt.subplots(3,1)
    font = {'fontname': 'serif'}
    plt.rcParams.update({'font.family':'serif'})
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig.suptitle(f"FFNN Test Performance \n LR={lr}, h-Neurons={neurons}, h-Layers={hidden_layers}, Steps={n_steps_history}", **font, y=1.02)
    ax[0].plot(time_steps, test_loss_l2, linewidth=1.0, c="b", label=r"$L_2$ loss")
    ax[1].plot(time_steps, test_loss_lmax, linewidth=1.0, c="g", label=r"$L_{max}$ loss")
    ax[2].plot(time_steps, r2score, linewidth=1.0, c="r", label="R² score")
    
    ax[2].set_xlabel("t [s]", **font)
    
    ax[2].get_shared_x_axes().join(ax[0], ax[1], ax[2])
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])

    for axis in ax:
        for tick in axis.get_xticklabels():
            tick.set_fontname("serif")
        for tick in axis.get_yticklabels():
            tick.set_fontname("serif")

    ax[0].set_ylabel(r"$L_2$ loss", **font)
    ax[1].set_ylabel(r"$L_{max}$ loss", **font)
    ax[2].set_ylabel("R² score", **font)
    
    ax[0].get_yaxis().set_label_coords(-0.14,0.5)
    ax[1].get_yaxis().set_label_coords(-0.14,0.5)
    ax[2].get_yaxis().set_label_coords(-0.14,0.5)
    
    # ax[0].legend(prop={'size': 9, 'family':'serif'})#, loc="right")
    # ax[1].legend(prop={'size': 9, 'family':'serif'})#, loc="right")
    # ax[2].legend(prop={'size': 9, 'family':'serif'})#, loc="right")
    
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    
    ax_handles = []
    ax_labels = []
    for axis in ax:
        handle, label = axis.get_legend_handles_labels()
        ax_handles.extend(handle)
        ax_labels.extend(label)
        
    fig.legend(ax_handles, ax_labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    fig.savefig(f"{save_plots_in}/evalutation_lr{lr}_neurons{neurons}_nlayers{hidden_layers}_nhistory{n_steps_history}.svg", bbox_inches="tight")

    # Plotting for cd and cl predicitons
    fig, ax = plt.subplots(2,1)
    fig.suptitle(f"FFNN Test Prediction \n LR={lr}, Neurons={neurons}, Hidden Layers={hidden_layers}, Steps={n_steps_history}", **font)

    ax[0].plot(time_steps, prediction_cd, label=r"pred. drag $c_D$", c="C3", ls="--", linewidth=1.0)
    ax[0].plot(time_steps, labels_cd, label=r"real drag $c_D$", c="k", linewidth=1.0)
    ax[1].plot(time_steps, prediction_cl, label=r"pred. lift $c_L$", c="C3", ls="--", linewidth=1.0)
    ax[1].plot(time_steps, labels_cl, label=r"real lift $c_L$", c="k", linewidth=1.0)

    ax[1].set_xlabel("t [s]", **font)

    ax[1].get_shared_x_axes().join(ax[0], ax[1])
    ax[0].set_xticklabels([])
    
    for axis in ax:
        for tick in axis.get_xticklabels():
            tick.set_fontname("serif")
        for tick in axis.get_yticklabels():
            tick.set_fontname("serif")

    ax[0].set_ylabel(r"$c_D$", **font)
    ax[1].set_ylabel(r"$c_L$", **font)

    ax[0].get_yaxis().set_label_coords(-0.1,0.5)
    ax[1].get_yaxis().set_label_coords(-0.1,0.5)

    ax[0].legend(prop={'size': 9, 'family':'serif'})# loc="upper right"
    ax[1].legend(prop={'size': 9, 'family':'serif'})# loc="upper right"
    
    ax[0].grid()
    ax[1].grid()
    
    fig.tight_layout()
    fig.savefig(f"{save_plots_in}/cdclprediction_lr{lr}_neurons{neurons}_nlayers{hidden_layers}_nhistory{n_steps_history}.svg", bbox_inches="tight")

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
    fig.savefig(f"{save_plots_in}evalutation_lr{lr}_neurons{neurons}_nlayers{hidden_layers}_nhistory{n_steps_history}.svg", bbox_inches="tight")
    
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
    fig.savefig(f"{save_plots_in}cdclprediction_lr{lr}_neurons{neurons}_nlayers{hidden_layers}_nhistory{n_steps_history}.svg", bbox_inches="tight")
