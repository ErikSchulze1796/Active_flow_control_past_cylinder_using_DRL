import torch as pt

import numpy as np

from matplotlib import pyplot as plt

def plot_loss(train_loss, val_loss, loss_type: str(), lr: float(), save_plots_in: str(), show: bool=False):
    epochs = len(train_loss)
    
    fig, ax = plt.subplots()
    ax.set_title(r"Epoch vs. {} loss; Learning rate = {}".format(loss_type, lr))
    ax.plot(range(1, epochs+1), train_loss, lw=1.0, label="training loss")
    ax.plot(range(1, epochs+1), val_loss, lw=1.0, label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{loss_type} loss")
    ax.set_xlim(1, epochs+1)
    ax.set_yscale("log")
    fig.legend()
    if show:
        fig.show()
    fig.savefig(f"{save_plots_in}/train_val_loss_lr{lr}.svg", bbox_inches="tight")

def plot_coefficient_prediction(time_steps, reference, prediction, y_label: str(), save_plots_in: str(), lr: float()):
    assert reference.shape == prediction.shape, "Dimensions of <reference> and <prediction> are not matching"
    assert reference.shape == time_steps.shape, "Dimensions of <reference> and <time_steps> are not matching"

    fig, ax = plt.subplots()
    ax.set_title(r"Time vs. ${}$; Learning rate = {}".format(y_label, lr))
    ax.plot([], label="reference", c="k")
    ax.plot([], label="prediction", c="C3", ls="--")
    ax.plot(time_steps, reference, c="k")
    ax.plot(time_steps, prediction, ls="--", c="C3")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"${}$".format(y_label))
    ax.set_xlim(0.0, max(time_steps))
    ax.set_xticks(np.arange(0.0, round(max(time_steps)+1), 1.0))
    
    # y_max = max(reference)
    # y_min = min(reference)
    # # Set min max values for y axis scaling
    # if min(prediction) < y_min:
    #     y_min = min(prediction)
    # if max(prediction) > y_max:
    #     y_max = max(prediction)
        
    # plt.ylim(y_min, y_max)
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=[0.5, 1.12])
    fig.savefig(f"{save_plots_in}/{y_label}_coefficient_pred_vs_org_lr{lr}.svg", bbox_inches="tight")
