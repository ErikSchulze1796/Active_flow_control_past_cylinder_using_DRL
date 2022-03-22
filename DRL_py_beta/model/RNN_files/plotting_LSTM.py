import pandas as pd
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    fig.savefig(f"{save_plots_in}train_val_loss_lr{lr}_neurons{n_neurons}_nlayers{n_layers}_nhistory{n_steps_history}.svg", bbox_inches="tight")

def plot_evaluation_multiple_trajectories(time_steps,
                                          l2,
                                          lmax,
                                          r2,
                                          save_plots_in: str,
                                          lr: float,
                                          neurons: int,
                                          hidden_layers: int,
                                          n_steps_history: int,
                                          rows: int,
                                          columns: int=3):
    """_summary_

    Parameters
    ----------
    time_steps : _type_
        Time steps of trajecotries
    l2 : _type_
        L2 loss for test trajectories
    lmax : _type_
        Lmax loss for test trajectories
    r2 : _type_
        R2 score for test trajectories
    save_plots_in : str
        Path for saving plots
    lr : float
        Learning rate
    neurons : int
        Number of neurons per hidden layer
    hidden_layers : int
        Number of hidden layers
    n_steps_history : int
        Number of time steps used for prediction / Sliding window size
    rows : int
        Number of rows for subplots
    columns : int, optional
        Number of columns for subplots, by default 3
    """
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

        ax[idx][0].grid()
        ax[idx][1].grid()
        ax[idx][2].grid()
        ax[idx][0].set_xlabel(r"$t^{*}$", **font)
        ax[idx][1].set_xlabel(r"$t^{*}$", **font)
        ax[idx][2].set_xlabel(r"$t^{*}$", **font)

    # Set tick labels to serif
    for axis1 in ax:
        for axis2 in axis1:
            for tick in axis2.get_xticklabels():
                tick.set_fontname("serif")
            for tick in axis2.get_yticklabels():
                tick.set_fontname("serif")

    # Get axis labels for having only one legend
    ax_handles = []
    ax_labels = []
    for axis2 in ax[0]:
        handle, label = axis2.get_legend_handles_labels()
        ax_handles.extend(handle)
        ax_labels.extend(label)

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.5)

    fig.legend(ax_handles, ax_labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=3)
    fig.savefig(f"{save_plots_in}evalutation_lr{lr}_neurons{neurons}_nlayers{hidden_layers}_nhistory{n_steps_history}.svg", bbox_inches="tight")

def plot_coefficient_predictions_multiple_trajs(time_steps,
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
                                columns: int=2):
    """Plot cD and cL values for multiple trajectories

    Parameters
    ----------
    time_steps :array
        Time steps for trajectories
    labels_cd : array
        cD ground truth values
    labels_cl : array
        cL ground truth values
    predictions_cd : array
        Predictions for cD
    predictions_cl : array_
        Predictions for cL
    save_plots_in : str
        Save plots to path
    lr : float
        Learning rate
    neurons : int
        Number of hidden neurons per layer
    hidden_layers : int
        Number of hidden layers
    n_steps_history : int
        Number of time steps used for prediction / Sliding window size
    rows : int
        Number of rows for subplots
    columns : int, optional
        Number of columns for subplots, by default 2
    """
    # Plotting for cd and cl predicitons
    font = {'fontname': 'serif'}
    plt.rcParams.update({'font.family':'serif'})
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig, ax = plt.subplots(rows,columns)
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

    # set axis tick labels to serif
    for axis1 in ax:
        for axis2 in axis1:
            for tick in axis2.get_xticklabels():
                tick.set_fontname("serif")
            for tick in axis2.get_yticklabels():
                tick.set_fontname("serif")

    # Get axis labels for having only one legend
    ax_handles = []
    ax_labels = []
    for axis2 in ax[0]:
        handle, label = axis2.get_legend_handles_labels()
        ax_handles.extend(handle)
        ax_labels.extend(label)

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.28, hspace=0.5)

    fig.legend(ax_handles, ax_labels, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=2)
    fig.savefig(f"{save_plots_in}cdclprediction_lr{lr}_neurons{neurons}_nlayers{hidden_layers}_nhistory{n_steps_history}.svg", bbox_inches="tight")

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
    """Plots multiple heat maps of the pressure per sensor and per time step of trajectory

    Parameters
    ----------
    time_steps : array
        Time steps for trajectories
    reference : array
        Ground truth pressure values
    predictions : array
        Pressure predictions
    save_plots_in : str
        Save plots to path
    lr : float
        Learning rate
    neurons : int
        Number of hidden neurons per layer
    hidden_layers : int
        Number of hidden layers
    n_steps_history : int
        Number of time steps used for prediction / Sliding window size
    rows : int
        Number of rows for subplots
    columns : int
        Number of columns for subplots
    """

    
    fig, axes = plt.subplots(rows, columns)
    faxes = axes.ravel()
    fig.suptitle(f"FFNN Test: Relative Pressure Error per Sensor vs. Time \n LR={lr}, h-Neurons={n_neurons}, h-Layers={n_layers}, Steps={n_steps_history}")
    plt.rcParams.update({'font.family':'serif'})
    # Load sensor positions in polar coordinates
    sensor_positions = pd.read_csv("plotting/centers.csv", sep=" ", header=0)

    p_max_error = 0
    for idx, time_steps_traj in enumerate(time_steps):
        error_rel = ((reference[idx] - predictions[idx]) / 2).abs() * 100
        max_error = torch.max((error_rel))
        if max_error > p_max_error:
            p_max_error = max_error

    for idx, time_steps_traj in enumerate(time_steps):
        error_rel = ((reference[idx] - predictions[idx]) / 2).abs() * 100
        # Sort sensor errors by position using the polar angle
        for i, row in enumerate(error_rel):
            error_rel[i,:] = torch.Tensor([x for _, x in sorted(zip(torch.from_numpy(sensor_positions.polar.values), row), key=lambda pair: pair[0])])
        
        # Transpose since colormesh is transposed
        error_rel = torch.transpose(error_rel, 0, 1)
        sensors = torch.linspace(1, 400, 400)
        
        pcol = faxes[idx].pcolormesh(time_steps_traj, sensors, error_rel, shading='auto', vmin=0, vmax=p_max_error)

        divider = make_axes_locatable(faxes[idx])
        cax = divider.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(pcol, cax=cax, label="Rel. Error [%]")
        
        faxes[idx].set_xlabel(r"$t^{*}$")
        faxes[idx].set_ylabel(r"Sensor")
    
    if (len(predictions) % 2) != 0:
        fig.delaxes(faxes[-1])

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=0.5)

    fig.savefig(f"{save_plots_in}relative_pressure_error_time_heatmap_lr{lr}_neurons{n_neurons}_nlayers{n_layers}_nhistory{n_steps_history}.png", bbox_inches="tight")

