import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Staley2017Model(nn.Module):
    """
    Staley et al. (2017) logistic regression model:
    p = 1 / (1 + exp(-(B + Ct*T*R + Cf*F*R + Cs*S*R)))
    """
    
    def __init__(self, features, duration='15min'):
        super().__init__()
        self.duration = duration
        self.name = 'logres'
        
        # Feature indices for T, F, S
        self.T_idx = features.index('PropHM23')
        self.F_idx = features.index('dNBR/1000')
        self.S_idx = features.index('KF') 
        
        # Rainfall index based on duration
        self.R_idx = {
            '15min': features.index('Acc015_mm'), 
            #'30min': features.index('Acc030_mm'), 
            #'60min': features.index('Acc060_mm')
        }[duration]
        
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
        return torch.sigmoid(logit)


