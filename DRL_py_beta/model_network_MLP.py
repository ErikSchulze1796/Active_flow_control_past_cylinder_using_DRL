"""
This file contains the module for the multi-layer perceptron model network
"""

import torch
from torch import nn
import torch.nn.functional as F

class FFMLP(nn.Module):
    """
    This class contains the multi-layer perceptron implementation for the pressure prediction from time t to t+1
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.n_inputs = kwargs.get("n_inputs", 401)
        self.n_outputs = kwargs.get("n_outputs", 402)
        self.n_layers = kwargs.get("n_layers", 2)
        self.n_neurons = kwargs.get("n_neurons", 100)
        self.activation = kwargs.get("activation", torch.relu)
        self.layers = torch.nn.ModuleList()
        # input layer to first hidden layer
        self.layers.append(torch.nn.Linear(self.n_inputs, self.n_neurons))
        # add more hidden layers if specified
        if self.n_layers > 1:
            for hidden in range(self.n_layers-1):
                self.layers.append(torch.nn.Linear(
                    self.n_neurons, self.n_neurons))
        # last hidden layer to output layer
        self.layers.append(torch.nn.Linear(self.n_neurons, self.n_outputs))
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        
    def forward(self, x):
        """
        This function handles the feed forwarding in the neural network(NN)
        ----------
        Args:
            x: Array or tensor containing current values of surface pressure from "sensors" at time t
            and current control action at time t
        ----------
        Returns:
            p_t1: Array or tensor containing predicted values of surface pressure from
            sensors" as well as drag and lift coefficientsat time t+1
            
        """
        for i_layer in range(len(self.layers)-1):
            x = self.activation(self.layers[i_layer](x))
        return self.layers[-1](x)
    
