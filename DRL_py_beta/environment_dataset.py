import torch as pt
from torch.utils.data import Dataset

class EnvironmentStateDataset(Dataset):
    def __init__(self, features_file, labels_file):
        self.state_features = pt.load(features_file)
        self.state_labels = pt.load(labels_file)

    def __len__(self):
        return len(self.state_labels)

    def __getitem__(self, idx):
        state = self.state_features[idx]
        label = self.state_labels[idx]
        
        return state, label
    