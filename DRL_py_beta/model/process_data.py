from cProfile import label
from xml.sax.handler import feature_external_ges
import torch

def load_trajectory_data(file_path: str, split: bool=False):
    trajectory_data = torch.load(file_path)
    
    if split:
        times = trajectory_data[:,:,0]
        n_pressure = trajectory_data[:,:,1:-3]
        c_d = trajectory_data[:,:,-3]
        c_l = trajectory_data[:,:,-2]
        omega = trajectory_data[:,:,-1]
    
        return times, n_pressure, c_d, c_l, omega
    
    else:
        return trajectory_data

def sliding_windows_single_trajectory(data, seq_length_features, seq_length_labels):
    
    assert seq_length_features + seq_length_labels <= data.shape[0], "Cannot generate a feature label pair if feature and label sequences sizes are larger than data size"
    x = torch.zeros(data.shape[0]-seq_length_features-(seq_length_labels-1), seq_length_features, data.shape[-1])
    y = torch.zeros(data.shape[0]-seq_length_features-(seq_length_labels-1), seq_length_labels, data.shape[-1])

    for i in range(data.shape[0]-seq_length_features-(seq_length_labels-1)):
        _x = data[i:(i+seq_length_features), :]
        _y = data[i+seq_length_features:(i+seq_length_features+seq_length_labels), :]
        x[i] = _x
        y[i] = _y

    return x, y

def create_random_feature_label_data(data, seq_length_features, seq_length_labels):
    features = torch.zeros(data.shape[0]*(data.shape[1]-seq_length_features-(seq_length_labels-1)), seq_length_features, 401)
    labels = torch.zeros(data.shape[0]*(data.shape[1]-seq_length_features-(seq_length_labels-1)),seq_length_labels, 402)
    
    for i, traj in enumerate(data):
        
        features_traj, labels_traj = sliding_windows_single_trajectory(traj, seq_length_features, seq_length_labels)
        features_traj = torch.cat((features_traj[:,:,1:-3],features_traj[:,:,-1].unsqueeze(dim=2)), dim=2)
        labels_traj = labels_traj[:,:,1:-1]
        features[(i*len(features_traj)):((i+1)*len(features_traj)),:,:] = features_traj
        labels[(i*len(labels_traj)):((i+1)*len(labels_traj)),:,:] = labels_traj
        
    features, labels, _ = shuffle_data(features, labels)
    
    return features, labels

def set_n_pressure_sensors(data, n_keep: int, equidistant: bool=True, idx_keep: object=None):
    if equidistant:
        assert ((data.shape[-1]-1) % n_keep) == 0, "You can only keep a number of sensors that is divisible without residue."
        keep_every_nth = int(data.shape[-1] / n_keep)
        transformed_data = torch.cat((data[:,:,:-1:keep_every_nth], data[:,:,-1].unsqueeze(dim=2)),dim=2)
    else:
        transformed_data = torch.cat((data[:,:,idx_keep], data[:,:,-1].unsqueeze(dim=2)),dim=2)
    
    return transformed_data

def shuffle_data(features, labels, seed=0):
    torch.manual_seed(seed)
    idxes = torch.randperm(features.shape[0])
    features_shuffled = features[idxes]
    labels_shuffled = labels[idxes]
    
    return features_shuffled, labels_shuffled, idxes

class MinMaxScaler(object):
    """Class to scale/re-scale data to the range [-1, 1] and back by min/max scaling.
    """
    def __init__(self):
        """Constructor of MinMaxScaler class
        """
        self.min = None
        self.max = None
        self.trained = False
        self.p_min = -1.6963981
        self.p_max = 2.028614
        self.c_d_min = 2.9635367
        self.c_d_max = 3.4396918
        self.c_l_min = -1.8241948
        self.c_l_max = 1.7353026
        self.omega_min = -9.999999
        self.omega_max = 10.0
    
    def scale_features(self, data):
        assert len(data.shape) == 3, f"Expected data to have dimension of 3, got {len(data.shape)}"
        data_norm = torch.zeros(data.shape)
        data_norm[:,:,:-1] = (data[:,:,:-1] - self.p_min) / (self.p_max - self.p_min)
        data_norm[:,:,-1] = (data[:,:,-1] - self.omega_min) / (self.omega_max - self.omega_min)
        return 2.0*data_norm - 1.0 # Scale between [-1, 1]
    
    def scale_labels(self, data):
        assert len(data.shape) == 3, f"Expected data to have dimension of 3, got {len(data.shape)}"
        data_norm = torch.zeros(data.shape)
        data_norm[:,:,:-2] = (data[:,:,:-2] - self.p_min) / (self.p_max - self.p_min)
        data_norm[:,:,-2] = (data[:,:,-2] - self.c_d_min) / (self.c_d_max - self.c_d_min)
        data_norm[:,:,-1] = (data[:,:,-1] - self.c_l_min) / (self.c_l_max - self.c_l_min)
        return 2.0*data_norm - 1.0 # Scale between [-1, 1]

    def rescale_features(self, data_norm):
        assert len(data_norm.shape) == 3, f"Expected data to have dimension of 3, got {len(data_norm.shape)}"
        data = torch.zeros(data_norm.shape)
        data = (data_norm + 1.0) * 0.5
        data[:,:,:-1] = data[:,:,:-1] * (self.p_max - self.p_min) + self.p_min
        data[:,:,-1] = data[:,:,-1] * (self.omega_max - self.omega_min) + self.omega_min
        return data
    
    def rescale_labels(self, data_norm):
        assert len(data_norm.shape) == 3, f"Expected data to have dimension of 3, got {len(data_norm.shape)}"
        data = torch.zeros(data_norm.shape)
        data = (data_norm + 1.0) * 0.5
        data[:,:,:-2] = data[:,:,:-2] * (self.p_max - self.p_min) + self.p_min
        data[:,:,-2] = data[:,:,-2] * (self.c_d_max - self.c_d_min) + self.c_d_min
        data[:,:,-1] = data[:,:,-1] * (self.c_l_max - self.c_l_min) + self.c_l_min
        return data

from torch.utils.data import Dataset

class EnvironmentStateDataset_from_variable(Dataset):
    def __init__(self, features, labels):
        self.state_features = features
        self.state_labels = labels

    def __len__(self):
        return len(self.state_labels)

    def __getitem__(self, idx):
        state = self.state_features[idx]
        label = self.state_labels[idx]
        
        return state, label

def eval_multistep_model(decoder, encoder, features_norm, labels_norm, n_p_keep=400):
    encoder.eval()
    decoder.eval()
    predictions = torch.zeros(labels_norm.shape[0],1,labels_norm.shape[2])
    keep_every_nth = int(features_norm.shape[-1] / n_p_keep)


    pred_from = torch.zeros(features_norm.shape)
    pred_from[0] = features_norm[0]
    pred_from[:,:,-1] = features_norm[:,:,-1]
    for i, seq in enumerate(pred_from):
        hidden = encoder(seq.unsqueeze(dim=0))
        output_seq = decoder(labels_norm[i].unsqueeze(dim=0), hidden)
        predictions[i] = output_seq[:,0,:].detach()
        if i == features_norm.shape[0]-1:
            break
        pred_from[i+1,:-1,:] = pred_from[i,1:,:]
        pred_from[i+1,-1,:-1] = output_seq[:,0,:-2:n_p_keep].detach()

    return predictions

from sklearn.metrics import r2_score
def eval_model_metrics(predictions, labels):
    
    test_loss_l2 = torch.zeros(labels.shape[0])
    test_loss_lmax = torch.zeros(labels.shape[0])
    r2score = torch.zeros(labels.shape[0])

    for i, pred in enumerate(predictions):
        test_loss_l2[i] = (pred - labels[i]).square().mean()
        test_loss_lmax[i] = (pred - labels[i]).absolute().max() / 2
        r2score[i] = r2_score(labels[i], pred)
            
        print("\r", "L2 loss: {:1.4f}, Lmax loss: {:1.4f}, r2 score: {:1.4f}".format(test_loss_l2[i], test_loss_lmax[i], r2score[i]), end="\r")

    return test_loss_l2, test_loss_lmax, r2score

def test():
    torch.manual_seed(0)
    location = "DRL_py_beta/training_pressure_model/initial_trajectories/data_train/initial_trajectory_data_train_set_pytorch_ep100_traj992_t-p400-cd-cl-omega.pt"
    data = load_trajectory_data(location, split=False)[:10]
    
    features, labels = create_random_feature_label_data(data, 4, 3)
    features = set_n_pressure_sensors(features, 16)

if __name__ == "__main__":
    test()
    
