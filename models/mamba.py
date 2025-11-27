import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm import Mamba
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MambaClassifier(nn.Module):
    def __init__(self, input_dim=16, d_model=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.duration = '15min'
        self.name = 'Mamba'

        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            #nn.Linear(d_model, d_model)
            for _ in range(n_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            #nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # For tabular data, we treat each sample as sequence of length 1
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Input projection
        x = self.input_proj(x)
        
        # Mamba layers
        for mamba_layer, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = mamba_layer(x)
            x = norm(x)
            x = self.dropout(x)
            x = residual + x  # Residual connection
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        return self.output_head(x).squeeze(-1)

class HybridMambaLogisticModel(nn.Module):
    def __init__(self, features, input_dim=16, d_model=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.duration = '15min'
        self.name = 'HybridMamba'
        
        self.rainfall_features = ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm']
        self.all_features = features
        
        # Get indices for rainfall vs non-rainfall features
        self.rainfall_indices = [self.all_features.index(feat) for feat in self.rainfall_features if feat in self.all_features]
        self.non_rainfall_indices = [i for i in range(len(self.all_features)) if i not in self.rainfall_indices]
        
        #print(f"Rainfall features ({len(self.rainfall_indices)}): {self.rainfall_features}")
        #print(f"Non-rainfall features ({len(self.non_rainfall_indices)}): {[self.all_features[i] for i in self.non_rainfall_indices]}")
        
        # Mamba pathway for non-rainfall features
        self.mamba_input_dim = len(self.non_rainfall_indices)
        self.mamba_input_proj = nn.Linear(self.mamba_input_dim, d_model)
        
        # Mamba backbone
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2,)
            #nn.Linear(d_model, d_model)
            for _ in range(n_layers)
        ])
        
        # Layer normalization for Mamba
        self.mamba_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        # Logistic regression pathway for rainfall features
        self.logistic_input_dim = len(self.rainfall_indices)
        self.logistic_layer = nn.Sequential(
            nn.Linear(self.logistic_input_dim, 1),
            #nn.Sigmoid()
        )
        
        # Combined output head
        self.combined_head = nn.Sequential(
            nn.Linear(d_model + 1, 32),  # +1 for logistic output
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            #nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def split_features(self, x):
        """Split input into rainfall and non-rainfall features"""
        # x shape: (batch_size, input_dim)
        non_rainfall_x = x[:, self.non_rainfall_indices]  # (batch_size, non_rainfall_dim)
        rainfall_x = x[:, self.rainfall_indices]          # (batch_size, rainfall_dim)
        return non_rainfall_x, rainfall_x
    
    def forward(self, x):
        non_rainfall_x, rainfall_x = self.split_features(x)
        
        mamba_out = self._mamba_forward(non_rainfall_x)
        logistic_out = self.logistic_layer(rainfall_x)  # (batch_size, 1)
        
        combined = torch.cat([mamba_out, logistic_out], dim=1)  # (batch_size, d_model + 1)
        
        # Final prediction
        output = self.combined_head(combined)
        return output.squeeze(-1)
    
    def _mamba_forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, mamba_input_dim)
        x = self.mamba_input_proj(x)  # (batch_size, 1, d_model)
        
        for mamba_layer, norm in zip(self.mamba_layers, self.mamba_norms):
            residual = x
            x = mamba_layer(x)
            x = norm(x)
            x = self.dropout(x)
            x = residual + x  # Residual connection
        
        x = x.mean(dim=1)  # (batch_size, d_model)
        return x