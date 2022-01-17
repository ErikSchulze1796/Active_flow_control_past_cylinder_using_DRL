"""Contains functions and classes for data preprocessing for the FFNN
    which is used as an environment model
"""

import torch as pt

class MinMaxScaler(object):
    """Class to scale/re-scale data to the range [-1, 1] and back by min/max scaling.
    """
    def __init__(self):
        """Constructor of MinMaxScaler class
        """
        self.min = None
        self.max = None
        self.trained = False

    def fit(self, data):
        """Stores min and max values of given data

        Parameters
        ----------
        data : torch.Tensor
            Data tensor
        """
        self.min = pt.min(data)
        self.max = pt.max(data)
        self.trained = True

    def refit(self, data):
        """Refits the min, max values if new data is available

        Parameters
        ----------
        data : torch.Tensor
            Data to get min max from
        """
        new_min = pt.min(data)
        new_max = pt.max(data)
        if new_min < self.min:
            self.min = new_min
        if new_max > self.max:
            self.max = new_max
        self.trained = True

    def scale(self, data):
        """Scales the data according to stored min/max values

        Parameters
        ----------
        data : torch.Tensor
            Data tensor

        Returns
        -------
        torch.Tensor
            Scaled/normalized data
        """
        assert self.trained
        # assert len(data.shape) == 3 # Assert dimension of input data
        data_norm = (data - self.min) / (self.max - self.min)
        return 2.0*data_norm - 1.0 # Scale between [-1, 1]

    def rescale(self, data_norm):
        """Rescales the normalized data back

        Parameters
        ----------
        data_norm : torch.Tensor
            Normalized data tensor

        Returns
        -------
        torch.Tensor
            Denormalized data
        """
        assert self.trained
        data = (data_norm + 1.0) * 0.5
        return data * (self.max - self.min) + self.min

def data_scaling(data):
    """Conducts input data preprocessing (i.e. min-max scaling, train, val, test set splitting etc.)
    Parameters
    ----------
    data : torch.Tensor
        Pytorch tensor containing the input data with the columns [t, p0-p400, cd, cl, omega]

    Returns
    -------
    data_norm: torch.Tensor
        Normalized data
    scaler_pressure: MinMaxScaler
        Scaler object for pressure min/max scaling
    scaler_cd: MinMaxScaler
        Scaler object for cd min/max scaling
    scaler_cl: MinMaxScaler
        Scaler object for cl min/max scaling
    scaler_omega: MinMaxScaler
        Scaler object for omega min/max scaling
    """
    # Normalize data
    data_norm = pt.Tensor(data)

    # Scale pressure values to min-max-scale but only using the training set
    scaler_pressure = MinMaxScaler()
    # Scale pressure data
    scaler_pressure.fit(data_norm[:, :,1:-3])
    data_norm[:, :, 1:-3] = scaler_pressure.scale(data_norm[:, :,1:-3])

    # Scale cd values to min-max-scale but only using the training set
    scaler_cd = MinMaxScaler()
    # Scale cd data
    scaler_cd.fit(data_norm[:, :,-3])
    data_norm[:, :,-3] = scaler_cd.scale(data_norm[:, :,-3].unsqueeze(dim=0))

    # Scale cl values to min-max-scale but only using the training set
    scaler_cl = MinMaxScaler()
    # Scale cl data
    scaler_cl.fit(data_norm[:, :,-2])
    data_norm[:, :,-2] = scaler_cl.scale(data_norm[:, :,-2].unsqueeze(dim=0))
    
    # Scale omega values to min-max-scale but only using the training set
    scaler_omega = MinMaxScaler()
    # Scale omega data
    scaler_omega.fit(data_norm[:, :,-1])
    data_norm[:, :,-1] = scaler_omega.scale(data_norm[:, :,-1].unsqueeze(dim=0))

    return data_norm, scaler_pressure, scaler_cd, scaler_cl, scaler_omega

def change_n_pressure_sensors(data_norm_p, every_nth_element: int):
    """Reduce the number of pressure sensors
        Keep Only the every nth sensor

    Parameters
    ----------
    data_norm_p : pt.Tensor
        All pressure data with dimension 400
    every_nth_element : int
        Keep only every nth sensor

    Returns
    -------
    pt.Tensor
        Reduced number of pressure sensor data
    """
    data_norm_p_reduced = pt.zeros((data_norm_p.shape[0], int(data_norm_p.shape[1]/every_nth_element)))
    for i, state in enumerate(data_norm_p):
        data_norm_p_reduced[i] = state[::every_nth_element]
    
    return data_norm_p_reduced

def generate_labeled_data(data_norm: pt.Tensor, n_steps_history: int, every_nth_element: int):
    """Creates the feature and label tensors from the input data

    Parameters
    ----------
    data_norm : pt.Tensor
        Normalized input data
    n_steps_history : int
        Number of subsequent states to be included in model input
    every_nth_element : int
        Every nth pressure sensor to be kept as input

    Returns
    -------
    [data_norm_features, data_norm_labels]: Dict(torch.Tensor, torch.Tensor)
        Dict of feature and label vector
    """
    reduced_p_dimension = int(n_steps_history * (((data_norm.shape[2] - 4) / every_nth_element) + 1))
    
    data_norm_features = pt.zeros(data_norm.shape[0],
                                  (data_norm.shape[1]-n_steps_history),
                                  reduced_p_dimension)
    data_norm_labels = pt.zeros(data_norm.shape[0],
                                (data_norm.shape[1]-n_steps_history),
                                (data_norm.shape[2]-2))
    
    # Features are all states of the time steps except for the last
    # Labels are all states of the time steps except for the first
    for i, trajectory in enumerate(data_norm):
        data_norm_features[i,:,:], data_norm_labels[i,:,:] = split_sequence(trajectory, n_steps_history, every_nth_element)

    return [data_norm_features, data_norm_labels]

def reshape_data(t: pt.Tensor, p: pt.Tensor, c_D: pt.Tensor, c_L: pt.Tensor, omega: pt.Tensor) -> pt.Tensor:
    """Create feature and label vectors.

    Parameters
    ----------
    t : pt.Tensor
        time steps
    p : pt.Tensor
        pressure from sensors along surface of cylinder
    c_D : pt.Tensor
        drag coefficient for time step t
    c_L : pt.Tensor
        lift coefficient for time step t
    omega : pt.Tensor
        rotation velocity which is the taken action by the DRL agent

    Returns
    -------
    pt.Tensor
        data suitable for training; the Tensor should have the shape (N_t, 404)
        corresponding to the states of the simulation containing time, omega, drag,
        lift and the pressure values of the sensors for a number N_t of time steps
    """
    assert t.shape[0] == p.shape[0]
    assert t.shape[0] == c_D.shape[0]
    assert t.shape[0] == c_L.shape[0]
    assert t.shape[0] == omega.shape[0]
    data = pt.zeros((t.shape[0], 4 + p.shape[1]))
    for i in range(t.shape[0]):
        data[i, 0] = t[i]
        data[i, 1:-3] = p[i][:]
        data[i, -3] = c_D[i]
        data[i, -2] = c_L[i]
        data[i, -1] = omega[i]
    return data

def split_sequence(data, n_steps_history: int, every_nth_element: int):
    """Splits the data into sequences of states for model input according to n_steps_history
    Based on: https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/

    Parameters
    ----------
    data : [type]
        Sequence that should be split to get features and labels from it for prediction
    n_steps_history : int
        Number of time steps to be included into the state vector
    every_nth_element : int
        Every nth pressure sensor to be kept as input

    Returns
    -------
    torch.Tensor
        current_t: Current state vector with possible previous states
        next_t: The label vector to be predicted

    """
    reduced_data = change_n_pressure_sensors(data[:,1:-3], every_nth_element)
    sequence_p_a = pt.cat((reduced_data, data[:,-1].unsqueeze(dim=1)), dim=1)
    # sequence_p_a = pt.cat((data[:,1:-3], data[:,-1].unsqueeze(dim=1)), dim=1)
    sequence_p_cdl = data[:,1:-1]
    
    current_t, next_t = pt.Tensor(pt.Tensor()), pt.Tensor(pt.Tensor())
    for i in range(len(sequence_p_a)):
        # Find the end of this pattern
        end_ix = i + n_steps_history
        # Check if we are beyond the sequence
        if end_ix > len(sequence_p_a)-1:
            break

        # Gather input and output parts of the pattern
        seq_current_t, seq_next_t = sequence_p_a[i:end_ix, :], sequence_p_cdl[end_ix:end_ix+1, :]
        current_t = pt.cat((current_t, seq_current_t.unsqueeze(dim=0)), 0)
        next_t = pt.cat((next_t, seq_next_t), 0)
        
    current_t = current_t.reshape((current_t.shape[0],current_t.shape[1]*current_t.shape[2]))

    return current_t, next_t

def split_data(data: pt.Tensor, val_portion_rel: float=0.10):
    """Splits data into train, validation and test set

    Parameters
    ----------
    data : pt.Tensor
        Input data to be split
    test_portion_rel : float, optional
        Test portion of the data, by default 0.20
    val_portion_rel : float, optional
        Validation portion of the data, by default 0.10

    Returns
    -------
    [train_data_features, train_data_labels], [val_data_features, val_data_labels], [test_data_features, test_data_labels]:
        Tuple(list(torch.Tensor, torch.Tensor), list(torch.Tensor, torch.Tensor), list(torch.Tensor, torch.Tensor))
        A tuple of three lists storing pairs of feature and label tensors for training, validation and test
    """
    val_portion_abs = round(val_portion_rel * data[0].shape[0])
    train_portion_abs = data[0].shape[0] - val_portion_abs
    #print(f"Absolute size of test set: {test_portion_abs}")
    #print(f"Absolute size of validation set: {val_portion_abs}")
    #print(f"Absolute size of training set: {train_portion_abs}")
    assert (val_portion_abs + train_portion_abs) == data[0].shape[0]


    # select snapshots for testing
    probs = pt.ones(data[0].shape[0])
    val_idx = pt.multinomial(probs, val_portion_abs)
    probs[val_idx] = 0.0
    train_idx = pt.multinomial(probs, train_portion_abs)
    #print("Testing snapshots: ", test_idx)
    #print("Validation snapshots: ", val_idx)
    #print("Training snapshots: ", train_idx)

    val_data_features = data[0][val_idx, :]
    val_data_labels = data[1][val_idx, :]
    train_data_features = data[0][train_idx, :]
    train_data_labels = data[1][train_idx, :]
    
    return [train_data_features, train_data_labels], [val_data_features, val_data_labels]
