import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import globals

class Staley2017Model(nn.Module):
    """
    Staley et al. (2017) logistic regression model:
    p = 1 / (1 + exp(-(B + Ct*T*R + Cf*F*R + Cs*S*R)))
    """
    
    def __init__(self, duration='15min'):
        super().__init__()
        self.duration = duration
        self.name = 'Staley'
        self.spatial = False
        
        # Feature indices for T, F, S
        self.T_idx = globals.all_features.index('PropHM23')
        self.F_idx = globals.all_features.index('dNBR/1000')
        self.S_idx = globals.all_features.index('KF') 
        
        # Rainfall index based on duration
        duration_map = {
            '15min': 'Acc015_mm',
            '30min': 'Acc030_mm',
            '60min': 'Acc060_mm'
        }

        feature_name = duration_map[duration]
        self.R_idx = globals.all_features.index(feature_name) if feature_name in globals.all_features else 0 # SETS TO 0 IF THAT FEATURE IS NOT PASSED IN

        # Initialize all parameters at 0
        #self.B = nn.Parameter(torch.tensor([0.0]))
        #self.Ct = nn.Parameter(torch.tensor([0.0]))
        #self.Cf = nn.Parameter(torch.tensor([0.0]))
        #self.Cs = nn.Parameter(torch.tensor([0.0]))
        
        # Create parameters
        self.B  = nn.Parameter(torch.empty(1))
        self.Ct = nn.Parameter(torch.empty(1))
        self.Cf = nn.Parameter(torch.empty(1))
        self.Cs = nn.Parameter(torch.empty(1))

        # Xavier/Glorot uniform initialization
        nn.init.xavier_uniform_(self.B.unsqueeze(0))
        nn.init.xavier_uniform_(self.Ct.unsqueeze(0))
        nn.init.xavier_uniform_(self.Cf.unsqueeze(0))
        nn.init.xavier_uniform_(self.Cs.unsqueeze(0))
    
    def forward(self, x, target=None):
        """
        Args:
            x: Full feature matrix (batch_size, num_features)
        """
        x = x[:, 0, :]

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
        return torch.sigmoid(logit), None

class LogisticRegression(nn.Module):
    def __init__(self, features, duration='15min'):
        super().__init__()
        self.duration = duration
        self.name = 'LogisticRegression'
        self.num_features = len(features)
        
        self.B = nn.Parameter(torch.empty(1)) # Intercept parameter
        self.C = nn.Parameter(torch.empty(self.num_features))
        
        nn.init.xavier_uniform_(self.B.unsqueeze(0))
        nn.init.xavier_uniform_(self.C.unsqueeze(0))
    
    def forward(self, x):
        logit = self.B + torch.sum(self.C * x, dim=1)   
        return torch.sigmoid(logit)