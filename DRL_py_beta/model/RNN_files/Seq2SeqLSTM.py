import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_dim: int, num_layers=1):
        """Constructor of encoder network

        Parameters
        ----------
        input_size : int
            Number of input features
        hidden_dim : int
            Number of hidden neurons
        num_layers : int, optional
            Number of hidden layers, by default 1
        """
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=self.num_layers, batch_first=True)

    def forward(self, x):
        """Forward function of encoder network

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of subsequent states

        Returns
        -------
        torch.Tensor
            Hidden state of encoder network
        """
        
        h_n = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim)
        c_n = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim)
        
        # Push through RNN layer (the output is irrelevant)
        _, (h_n, c_n) = self.lstm(x,(h_n, c_n))
        return (h_n, c_n)


class Decoder(nn.Module):
    """Decoder neural network class for time series prediction
    """

    def __init__(self, hidden_dim: int, output_size: int, num_layers=1):
        """Constructor function for the decoder

        Parameters
        ----------
        hidden_dim : int
            Number of hidden neurons
        output_size : int
            Number of output features
        num_layers : int, optional
            Number of hidden layers, by default 1
        """
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_size)

    def forward(self, labels, hidden):
        """Forward function of decoder

        Parameters
        ----------
        labels : torch.Tensor
            Labels tensor
        hidden : torch.Tensor
            Hidden state from decoder network

        Returns
        -------
        torch.Tensor
            Multistep output sequence
        """
        batch_size = labels.shape[0]
        num_steps = labels.shape[1]
        # Create initial start value/token for input sequence
        input = torch.zeros((labels.shape[0],1,labels.shape[-1]), dtype=torch.float)
        # Convert (batch_size, output_size) to (batch_size, seq_len, output_size)
        # input = input.unsqueeze(dim=1)
        output_seq = torch.zeros(labels.shape)

        for i in range(num_steps):
            # Push current input through LSTM
            output, hidden = self.lstm(input, hidden)
            # Push the output of last step through linear layer; returns (batch_size, feature_size)
            output = self.out(output[:,-1,:])
            # Save sequence data
            output_seq[:,i] = output
            # Generate input for next step by adding seq_len dimension (see above)
            input = output.unsqueeze(dim=1).detach()
            
        return output_seq
