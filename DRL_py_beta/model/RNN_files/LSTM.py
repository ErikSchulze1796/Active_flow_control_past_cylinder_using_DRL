import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMEnvironment(nn.Module):

    def __init__(self, input_size, hidden_size, layer_number, output_size, bidirectional, multi_step):
        super(LSTMEnvironment, self).__init__()
        
        # Output dimension
        self.output_size = output_size
        # Number of recurrences
        self.layer_number = layer_number
        # Input dimension
        self.input_size = input_size
        # Size of hidden state
        self.hidden_size = hidden_size
        # Is bidirectional
        self.bidirectional = bidirectional
        # Multi step prediction
        self.multi_step = multi_step
        
        # Step 1: LSTM
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=layer_number, batch_first=True, bidirectional=bidirectional)
        # Step2: the FNN
        if bidirectional: # we'll multiply the number of hidden neurons by 2
            self.layer = nn.Linear(hidden_size*2, output_size)
        else:
            self.layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.bidirectional:
            # Start state of hidden state
            h_0 = Variable(torch.zeros(
                self.layer_number*2, x.size(0), self.hidden_size))
            # Start state of cell state
            c_0 = Variable(torch.zeros(
                self.layer_number*2, x.size(0), self.hidden_size))
        else:
            # Start state of hidden state
            h_0 = Variable(torch.zeros(
                self.layer_number, x.size(0), self.hidden_size))
            # Start state of cell state
            c_0 = Variable(torch.zeros(
                self.layer_number, x.size(0), self.hidden_size))
    
        # Propagate input through LSTM
        output, (last_hidden_state, last_cell_state) = self.lstm(x, (h_0, c_0))
        
        # Return only last element of the sequence to get the new state
        output = output[:, -1, :]
        
        # Compute through last layer to get correct output dimension
        output = self.layer(output)
        
        return output
    
    @torch.jit.ignore
    def get_prediction(self, x):
        """Samples state prediction using the LSTM

        Parameters
        ----------
        x : torch.Tensor
            Last sequence of subsequent features

        Returns
        -------
        torch.Tensor
            Next state containing pressure, cD and cL
        """
        if self.bidirectional:
            # Start state of hidden state
            h_n = Variable(torch.zeros(
                self.layer_number*2, x.size(0), self.hidden_size))
            # Start state of cell state
            c_n = Variable(torch.zeros(
                self.layer_number*2, x.size(0), self.hidden_size))
        else:
            # Start state of hidden state
            h_n = Variable(torch.zeros(
                self.layer_number, x.size(0), self.hidden_size))
            # Start state of cell state
            c_n = Variable(torch.zeros(
                self.layer_number, x.size(0), self.hidden_size))

        output, (h_n_new, c_n_new) = self.lstm(x, (h_n, c_n))
        
        # Return only last element of the sequence to get the new state
        output = output[:, -1, :]
        
        # Compute through last layer to get correct output dimension
        output = self.layer(output)
        
        return output.detach()