"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.

	called in main.py
"""

import torch
from torch import nn
import torch.nn.functional as F

torch.set_default_tensor_type(torch.DoubleTensor)


class FCCA(nn.Module):
    """
        A standard in_dim-64-64-out_dim Feed Forward Neural Network for policy model(Actor).
    """

    def __init__(self, input_dim, hidden_dims, action_bounds):
        """
        The output layer dimensions are 2 neurons. 1st neuron in output layer represents mean value
        and 2nd neuron in output layer represents std value.

        Args:
            input_dim: input dimensions are equal to number of pressure sensors
                        sensors -> p values of patches at surface of cylinder.
            hidden_dims: 64x64
            activation_fc: neuron activation function = relu -> torch,nn.functional.F.relu
        """
        super(FCCA, self).__init__()
        self.linear_0 = torch.nn.Linear(input_dim, hidden_dims)
        self.linear_1 = torch.nn.Linear(hidden_dims, hidden_dims)
        self.linear_2 = torch.nn.Linear(hidden_dims, 2)
        self.env_min, self.env_max = action_bounds
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.env_min = torch.tensor(self.env_min,
                                    device=self.device, 
                                    dtype=torch.float32)

        self.env_max = torch.tensor(self.env_max,
                                    device=self.device, 
                                    dtype=torch.float32)
        self.nn_min = torch.Tensor([float(0)]).to(self.device)
        self.nn_max = torch.Tensor([float(1)]).to(self.device)
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / (self.nn_max - self.nn_min) + self.env_min
        self.descale_fn = lambda x: (x - self.env_min) * (self.nn_max - self.nn_min) / (self.env_max - self.env_min) + self.nn_min


    def forward(self, x):
        """
        Feed forwarding in NN net

        Args:
            x: array or tensor containing value of pressure of patches at the surface of cylinder
                    must be (n_sensors x 1) dimensions.

        Returns:
            x_mean: mean value from 1st neuron of output layer, mean value of taken action.
            x_std: std value from 2nd neuron of output layer, std value of taken action.

        """
        # feed forwards to layers
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        return 1.0 + F.softplus(self.linear_2(x))
    
    @torch.jit.ignore
    def get_predictions(self, states, actions):
        """
        To compute log probability of taken action and entropy of taken action from the distribution.
        Args:
            states: input array, pressure array
            actions: action array
        Returns: tensors of log probability and tensor of entropy
        """

        # get mean and std of action for the supplied state
        output_layer = self.forward(torch.from_numpy(states))
        alpha, beta = output_layer[:, :, 0], output_layer[:, :, 1]
        alpha = alpha.squeeze()
        beta = beta.squeeze()
        # get distribution from mean and std by feed forward
        dist = torch.distributions.Beta(alpha, beta)
        d_actions = self.descale_fn(torch.from_numpy(actions))
        # compute log probabilities and entropy
        logpas = dist.log_prob(d_actions)
        entropies = dist.entropy()

        return logpas, entropies

    @torch.jit.ignore
    def select_action(self, states):
        """Samples an action given the states using the policy network

        Parameters
        ----------
        states : numpy.array
            400 pressure values

        Returns
        -------
        float
            Action sampled from the policy network
        """
        # Get alpha and beta coefficients of action for the supplied state
        output_layer = self.forward(torch.from_numpy(states))
        alpha, beta = output_layer[:,:,0], output_layer[:,:,1]
        alpha = alpha.squeeze()
        beta = beta.squeeze()
        # Get beta distribution
        dist = torch.distributions.Beta(alpha, beta)
        # Sample action from distribution
        action_pre_scale = dist.sample().detach().cpu().item()
        action = self.rescale_fn(action_pre_scale)
        
        action_mean_pre_scale = dist.mean.detach().cpu().item()
        action_mean = self.rescale_fn(action_mean_pre_scale)
        
        action_log_std_pre_scale = dist.stddev.detach().cpu().item()
        # Still unscaled to match previous model output
        action_log_std = action_log_std_pre_scale #self.rescale_fn(action_log_std_pre_scale)
        
        return action, action_mean, action_log_std, alpha, beta
    

class FCV(nn.Module):
    """
        A standard in_dim-64-64-out_dim Feed Forward Neural Network for value model(Critic).
    """

    def __init__(self, input_dim, hidden_dims):
        """
        The output layer dimensions are 1 neuron. The neuron in output layer represents value of the state.
        Args:
            input_dim: input dimensions are equal to number of pressure sensors
                        sensors -> p values of patches at surface of cylinder.
            hidden_dims: 64x64
            activation_fc: neuron activation function = relu -> torch,nn.functional.F.relu
        """
        super(FCV, self).__init__()
        self.linear_0 = torch.nn.Linear(input_dim, hidden_dims)
        self.linear_1 = torch.nn.Linear(hidden_dims, hidden_dims)
        self.linear_2 = torch.nn.Linear(hidden_dims, 1)

    def forward(self, x):
        """
         Feed forwarding in NN net.

        Args:
            x: array or tensor containing value of pressure of patches at the surface of cylinder
                    must be (n_sensors x 1) dimensions.

        Returns: Tensor of an state value(value is pi_theta)

        """
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        return self.linear_2(x)
