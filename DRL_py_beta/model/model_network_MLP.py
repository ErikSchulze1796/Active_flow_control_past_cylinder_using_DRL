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

    def __init__(self, input_size=401, hidden_layer_size=64, output_size=402):
        """
        This class contains the multi-layer perceptron implementation for the pressure prediction from time t to t+1
        ----------
        Args:
            input_size: Number of input features/input layer size
            hidden_layer_size: Number of hidden layers
            output_size: Number of output features/output layer size
        """

        super(FFMLP, self).__init__()
        self.linear_0 = torch.nn.Linear(input_size, hidden_layer_size)
        self.linear_1 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear_2 = torch.nn.Linear(hidden_layer_size, output_size)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        
    def forward(self, x):
        """
        This function handles the feed forwarding in th neural network(NN)
        ----------
        Args:
            x: Array or tensor containing current values of surface pressure from "sensors" at time t
            and current control action at time t
        ----------
        Returns:
            p_t1: Array or tensor containing predicted values of surface pressure from
            sensors" as well as drag and lift coefficientsat time t+1
            
        """
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        return self.linear_2(x)
        