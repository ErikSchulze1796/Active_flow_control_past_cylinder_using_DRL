"""
This file contains the module for the multi-layer perceptron model network
"""
import datetime

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
        # self.n_steps = kwargs.get("n_steps", 1)
        # self.n_p_sensors = kwargs.get("n_p_sensors", 1)
        # self.p_min = kwargs.get("p_min")
        # self.p_max = kwargs.get("p_max")
        # self.omega_min = kwargs.get("omega_min")
        # self.omega_max = kwargs.get("omega_max")
        # self.c_d_min = kwargs.get("c_d_min")
        # self.c_d_max = kwargs.get("c_d_max")
        # self.c_l_min = kwargs.get("c_l_min")
        # self.c_l_max = kwargs.get("c_l_max")
        
        # self.normalization = False
        # if ((self.p_min is not None) and
        #     (self.p_max is not None) and
        #     (self.omega_min is not None) and
        #     (self.omega_max is not None) and
        #     (self.c_d_min is not None) and
        #     (self.c_d_max is not None) and
        #     (self.c_l_min is not None) and
        #     (self.c_l_max is not None)
        #     ):
        #     self.normalization = True

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

    # def forward1(self, x):
    #     """
    #     This function handles the feed forwarding in the neural network(NN)
    #     ----------
    #     Args:
    #         x: Array or tensor containing current values of surface pressure from "sensors" at time t
    #         and current control action at time t
    #     ----------
    #     Returns:
    #         p_t1: Array or tensor containing predicted values of surface pressure from
    #         sensors" as well as drag and lift coefficientsat time t+1
            
    #     """
    #     x = self.scale(x)
    #     for i_layer in range(len(self.layers)-1):
    #         x = self.activation(self.layers[i_layer](x))
    #     return self.rescale(self.layers[-1](x))

    # def scale(self, data: torch.Tensor):
    #     """Scaling/normalization of the input data using min/max scaling to the range [-1,1]

    #     Parameters
    #     ----------
    #     data : torch.Tensor
    #         Input data to be normalized

    #     Returns
    #     -------
    #     torch.Tensor
    #         Min/max normalized input data
    #     """
    #     assert self.normalization, "Min/max values for input and output data must be set at network creation"
    #     start = datetime.datetime.now()
    #     data_norm = torch.zeros(data.shape)
    #     # Pressure normalization
    #     # If multiple time steps are used make sure to scale the correct values
    #     for step in range(self.n_steps):
    #         start_p = int(step * (self.n_p_sensors+1))
    #         end_p = int(start_p + self.n_p_sensors)
    #         data_norm[:,:,start_p:end_p] = (data[:,:,start_p:end_p] - self.p_min) / (self.p_max - self.p_min)
    #         # Omega normalization
    #         data_norm[:,:,end_p] = (data[:,:,end_p] - self.omega_min) / (self.omega_max - self.omega_min)
    #     data_norm = 2.0 * data_norm - 1.0 # Scale between [-1, 1]
    #     end = datetime.datetime.now()
    #     print("Scale time:")
    #     print(end-start)
    #     return data_norm
    
    # def rescale(self, data_norm: torch.Tensor):
    #     """Reverse min/max scaling of the input data from [-1,1]

    #     Parameters
    #     ----------
    #     data_norm : torch.Tensor
    #         Normalized input data to be rescaled

    #     Returns
    #     -------
    #     torch.Tensor
    #         Rescaled data that has been reverse min/max scaled
    #     """
    #     assert self.normalization, "Min/max values for input and output data must be set at network creation"
    #     start = datetime.datetime.now()
        
    #     data = torch.zeros(data_norm.shape)
        
    #     data_norm = (data_norm + 1.0) * 0.5
    #     # Pressure rescaling
    #     data[:,:,:-2] = data_norm[:,:,:-2] * (self.p_max - self.p_min) + self.p_min
    #     # c_d rescaling
    #     data[:,:,-2] = data_norm[:,:,-2] * (self.c_d_max - self.c_d_min) + self.c_d_min
    #     # c_l rescaling
    #     data[:,:,-1] = data_norm[:,:,-1] * (self.c_l_max - self.c_l_min) + self.c_l_min
    
    #     end = datetime.datetime.now()
    #     print("Rescale time:")
    #     print(end-start)
    #     return data

class WrapperModel(torch.nn.Module):
    def __init__(self, model, pmin, pmax, omegamin, omegamax, cdmin, cdmax, clmin, clmax, n_steps, n_sensors):
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
    
    def scale(self, x):
        
        for step in range(self._n_steps):
            start_p = int(step * (self._n_sensors+1))
            end_p = int(start_p + self._n_sensors)
            # Pressure scaling
            x[:,:,start_p:end_p] = (x[:,:,start_p:end_p] - self._pmin) / self._prange
            # Omega scaling
            x[:,:,end_p] = (x[:,:,end_p] - self._omegamin) / self._omegarange
            x = 2.0 * x - 1.0
            
        return x
        
    def rescale(self, x):
        x = (x + 1.0) * 0.5
        # Pressure rescaling
        x[:,:,:-2] = x[:,:,:-2] * self._prange + self._pmin
        # c_d rescaling
        x[:,:,-2] = x[:,:,-2] * self._cdrange + self._cdmin
        # c_l rescaling
        x[:,:,-1] = x[:,:,-1] * self._clrange + self._clmin

        return x
    
    def forward(self, x):
        x = self.scale(x)
        x = self._model(x)
        return self.rescale(x)