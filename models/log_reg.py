import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix

from eval import find_best_threshold, evaluate, compare_params

class Staley2017Model(nn.Module):
    """
    Staley et al. (2017) logistic regression model:
    p = 1 / (1 + exp(-(B + Ct*T*R + Cf*F*R + Cs*S*R)))
    """
    
    def __init__(self, duration='15min'):
        super().__init__()
        self.duration = duration
        self.name = 'logres'
        
        # Feature indices for T, F, S
        self.T_idx = 10  # PropHM23
        self.F_idx = 11  # dNBR/1000
        self.S_idx = 12  # KF
        
        # Rainfall index based on duration
        self.R_idx = {'15min': 13, '30min': 14, '60min': 15}[duration]
        
        # Initialize all parameters at 0
        self.B = nn.Parameter(torch.tensor([0.0]))
        self.Ct = nn.Parameter(torch.tensor([0.0]))
        self.Cf = nn.Parameter(torch.tensor([0.0]))
        self.Cs = nn.Parameter(torch.tensor([0.0]))
    
    def forward(self, x):
        """
        Args:
            x: Full feature matrix (batch_size, num_features)
        """
        # Extract the specific features we need
        T = x[:, self.T_idx]
        F = x[:, self.F_idx]
        S = x[:, self.S_idx]
        R = x[:, self.R_idx]
        
        B = self.B.squeeze()
        Ct = self.Ct.squeeze()
        Cf = self.Cf.squeeze()
        Cs = self.Cs.squeeze()
        
        logit = B + Ct * T * R + Cf * F * R + Cs * S * R
        # Clip for numerical stability
        logit = torch.clamp(logit, -500, 500)
        return torch.sigmoid(logit)


