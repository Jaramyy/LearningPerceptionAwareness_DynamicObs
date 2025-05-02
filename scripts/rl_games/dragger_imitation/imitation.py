# needed to import for allowing type-hinting:gym.spaces.Box | None
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np

CONFIG_IMITATION = {
    'architecture': 'feedforward',
    'lr': 0.001,
    'scheduler_factor': 1e-2,
    'scheduler_patience': 2,  # 2
    'scheduler_min_lr': 1e-4,
    'epochs': 5,
    'num_iterations' : 2000,
    "minibatch_size" : 32768,
    'batch_size': 32768,  # num_step = 2048, num_env = 4096
    'max_samples': 65536,
    'num_steps': 128,
    'best_checkpoint_path': './best_model.pth',
}



class StudentPolicy(nn.Module):
    def __init__(self, input_size=72, hidden_size=64, output_size=4):
        super(StudentPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.05)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.05)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(0.05)
        self.out = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()

    def forward(self, x):
        x1 = self.elu(self.fc1(x))
        x1 = self.dropout1(x1)
        
        x2 = self.elu(self.fc2(x1))
        x2 = self.dropout2(x2)
        
        x3 = self.elu(self.fc3(x2))
        x3 = self.dropout3(x3)
        
        out = self.out(x3)
        
        return out  # self.fc3(x)
    
    def loss(self):
        return nn.MSELoss()
    

# Define a custom Dataset for DAgger data
class DAggerDataset(Dataset):
    def __init__(self, data):
        self.data = data  # `data` is a list of (observation, expert_action) tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs, action = self.data[idx]
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)


class HDF5DAggerDataset(Dataset):
    def __init__(self, h5_path, device='cuda:0'):
        self.h5_path = h5_path
        self.device = device
        # We avoid keeping the file open to allow 'with' usage per access

        # Get length once to avoid re-opening for __len__
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['observations'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            obs = torch.tensor(f['observations'][idx], dtype=torch.float32, device=self.device)
            act = torch.tensor(f['actions'][idx], dtype=torch.float32, device=self.device)
        return obs, act
    
