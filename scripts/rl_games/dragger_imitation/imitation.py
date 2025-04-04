# needed to import for allowing type-hinting:gym.spaces.Box | None
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

CONFIG_IMITATION = {
    'architecture': 'feedforward',
    'lr': 0.001,
    'scheduler_factor': 1e-2,
    'scheduler_patience': 2,  # 2
    'scheduler_min_lr': 1e-4,
    'epochs': 10,
    'num_iterations' : 5000,
    'num_episodes': 100,
    'batch_size': 512,  #24576, num_step = 2048, num_env = 4096
    'max_samples': 32768,
}


class StudentPolicy(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, output_size=4):
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
