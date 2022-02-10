"""
This file contains the module for the multi-layer perceptron model network
"""
import datetime

import torch
from torch import nn
import torch.nn.functional as F

class FFNN(nn.Module):
    """
    This class contains the multi-layer perceptron implementation for the pressure prediction from time t to t+1
    """

    def __init__(self, **kwargs):
        """Constructor for the feedforward neural network (FFNN) class
        """
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

class WrapperModel(torch.nn.Module):
    def __init__(self, model, pmin, pmax, omegamin, omegamax, cdmin, cdmax, clmin, clmax, n_steps, n_sensors):
        """Constructor of WrapperModel class for min/max value initialization

        Parameters
        ----------
        model : torch.nn.Module
            Pytorch model for time series prediction
        pmin : float
            Minimal pressure value
        pmax : float
            Maximal pressure value
        omegamin : float
            Minimal omega (action) value
        omegamax : float
            Maximal omega (action) value
        cdmin : float
            Minimal drag coefficient
        cdmax : float
            Maximal drag coefficient
        clmin : float
            Minimal lift coefficient
        clmax : float
            Maximal lift coefficient
        n_steps : int
            Number of subsequent time steps used for prediction
        n_sensors : int
            Number of pressure sensors used for feature states
        """
        super(WrapperModel, self).__init__()
        self._model = model
        self._pmin = pmin
        self._pmax = pmax
        self._prange = pmax - pmin
        self._omegamin = omegamin
        self._omegamax = omegamax
        self._omegarange = omegamax - omegamin
        self._cdmin = cdmin
        self._cdmax = cdmax
        self._cdrange = cdmax - cdmin
        self._clmin = clmin
        self._clmax = clmax
        self._clrange = clmax - clmin
        self._n_sensors = n_sensors
        self._n_steps = n_steps
        self.input_scaled = True
        self.output_rescaled = True

    @torch.jit.ignore
    def scale(self, x):
        """Scales the input tensor to normalization range

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor

        Returns
        -------
        torch.Tensor
            Scaled feature tensor
        """
        x = x.clone()
        if self.input_scaled == False:
            return x
        
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)

        for step in range(self._n_steps):
            start_p = int(step * (self._n_sensors+1))
            end_p = int(start_p + self._n_sensors)
            # Pressure scaling
            x[:,start_p:end_p] = (x[:,start_p:end_p] - self._pmin) / self._prange
            # Omega scaling
            x[:,end_p] = (x[:,end_p] - self._omegamin) / self._omegarange
        x = 2.0 * x - 1.0
        
        return x

    @torch.jit.ignore
    def rescale(self, x):
        """Rescales the output from normalization range back

        Parameters
        ----------
        x : torch.Tensor
            Prediction tensor

        Returns
        -------
        torch.Tensor
            Rescaled prediction tensor
        """
        if self.output_rescaled == False:
            return x
        
        x = (x + 1.0) * 0.5
        # Pressure rescaling
        x[:,:-2] = x[:,:-2] * self._prange + self._pmin
        # c_d rescaling
        x[:,-2] = x[:,-2] * self._cdrange + self._cdmin
        # c_l rescaling
        x[:,-1] = x[:,-1] * self._clrange + self._clmin

        return x

    @torch.jit.ignore
    def input_scaling(self, input_scaled: bool):
        self.input_scaled = input_scaled

    @torch.jit.ignore
    def output_rescaling(self, output_rescaled: bool):
        self.output_rescaled = output_rescaled

    def forward(self, x):
        """Wrapper model forward function for scaling

        Parameters
        ----------
        x : torch.Tensor
            Unscaled input feature tensor

        Returns
        -------
        torch.Tensor
            Unscaled prediction tensor
        """
        x = self.scale(x)
        x = self._model(x)
        return self.rescale(x)