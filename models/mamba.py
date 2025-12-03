import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm import Mamba
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pdb

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
    


class HybridMambaLogisticModel_Mabel(nn.Module):
    def __init__(self, features, input_dim=16, d_model=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.duration = '15min'
        self.name = 'HybridMamba_Mabel'
        
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


        # Mamba pathway for rainfall features
        self.mamba_input_dim_rain = len(self.rainfall_indices)
        self.mamba_input_proj_rain = nn.Linear(self.mamba_input_dim_rain, d_model)
        
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

        # Combined output head
        self.combined_head_mamba = nn.Sequential(
            nn.Linear(d_model + d_model, 32),  # +1 for logistic output
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
        #pdb.set_trace()
        non_rainfall_x, rainfall_x = self.split_features(x)
        
        mamba_out = self._mamba_forward(non_rainfall_x)
        #logistic_out = self.logistic_layer(rainfall_x)  # (batch_size, 1)
        mamba_out_rain = self._mamba_forward_rain(rainfall_x)
        
        combined = torch.cat([mamba_out, mamba_out_rain], dim=1)  # (batch_size, d_model + 1)
        
        # Final prediction
        output = self.combined_head_mamba(combined)
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
    
    def _mamba_forward_rain(self, x):
        #pdb.set_trace()
        x = x.unsqueeze(1)  # (batch_size, 1, mamba_input_dim_rain)
        x = self.mamba_input_proj_rain(x)  # (batch_size, 1, d_model)
        
        for mamba_layer, norm in zip(self.mamba_layers, self.mamba_norms):
            residual = x
            x = mamba_layer(x)
            x = norm(x)
            x = self.dropout(x)
            x = residual + x  # Residual connection
        
        x = x.mean(dim=1)  # (batch_size, d_model)
        return x
    
import torch
import torch.nn as nn

# Assuming Mamba is imported from somewhere, e.g., from mamba_ssm import Mamba

# Mamba is a State Space Model (SSM) architecture.
# 

class ClusteredMambaModel_Flood(nn.Module):
    def __init__(self, features, input_dim=16, d_model=64, n_layers=4, dropout=0.1, n_clusters=1000):
        super().__init__()
        self.d_model = d_model
        self.name = 'ClusteredMamba_Flood'
        
        self.rainfall_features = ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm']
        self.all_features = features
        self.n_clusters = n_clusters
        
        # Get indices for rainfall vs non-rainfall features
        self.rainfall_indices = [self.all_features.index(feat) for feat in self.rainfall_features if feat in self.all_features]
        self.non_rainfall_indices = [i for i in range(len(self.all_features)) if i not in self.rainfall_indices]
        
        # --- 1. Clustering Projections ---
        # Instead of explicit clustering, we use a learned projection to a cluster space.
        
        # Non-Rainfall Clustering Projection
        self.non_rainfall_dim = len(self.non_rainfall_indices)
        if self.non_rainfall_dim > 0:
            self.non_rainfall_cluster_proj = nn.Linear(self.non_rainfall_dim, n_clusters)
            
        # Rainfall Clustering Projection
        self.rainfall_dim = len(self.rainfall_indices)
        if self.rainfall_dim > 0:
            self.rainfall_cluster_proj = nn.Linear(self.rainfall_dim, n_clusters)
        
        # --- 2. Mamba Pathways ---
        
        # Mamba pathway for non-rainfall (takes n_clusters input)
        self.mamba_input_proj_non_rain = nn.Linear(n_clusters, d_model)
        
        # Mamba pathway for rainfall (takes n_clusters input)
        self.mamba_input_proj_rain = nn.Linear(n_clusters, d_model)
        
        # Mamba backbone (shared)
        # Note: Mamba (SSM) layers excel at capturing sequential/temporal dependencies.
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        # Layer normalization for Mamba
        self.mamba_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        # --- 3. Combined Output Head ---
        # Combined output: Mamba output (d_model) + Mamba_Rain output (d_model)
        self.combined_head = nn.Sequential(
            nn.Linear(d_model + d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def split_features(self, x):
        """Split input into rainfall and non-rainfall features"""
        # x shape: (batch_size, num_features)
        non_rainfall_x = x[:, self.non_rainfall_indices]
        rainfall_x = x[:, self.rainfall_indices]
        return non_rainfall_x, rainfall_x
    
    def _mamba_forward(self, x, input_proj):
        # x shape: (batch_size, n_clusters)
        
        # Mamba requires an input sequence, so we unsqueeze to create a sequence of length 1
        x = x.unsqueeze(1)  # (batch_size, 1, n_clusters)
        x = input_proj(x)   # (batch_size, 1, d_model)
        
        for mamba_layer, norm in zip(self.mamba_layers, self.mamba_norms):
            residual = x
            x = mamba_layer(x)
            x = norm(x)
            x = self.dropout(x)
            # The residual connection is essential in deep networks.
            x = residual + x  # Residual connection
        
        x = x.mean(dim=1)  # Pool across the sequence dimension (length 1) -> (batch_size, d_model)
        return x
    
    def forward(self, x):
        non_rainfall_x_raw, rainfall_x_raw = self.split_features(x)
        
        # 1. Clustering Step (Projection)
        # Non-Rainfall: (batch_size, non_rainfall_dim) -> (batch_size, n_clusters)
        non_rainfall_clustered = self.non_rainfall_cluster_proj(non_rainfall_x_raw)
        
        # Rainfall: (batch_size, rainfall_dim) -> (batch_size, n_clusters)
        rainfall_clustered = self.rainfall_cluster_proj(rainfall_x_raw)
        
        # 2. Mamba Forward Pass on Clustered Data
        # Use the same Mamba layers for both, but different input projections.
        mamba_out_non_rain = self._mamba_forward(non_rainfall_clustered, self.mamba_input_proj_non_rain)
        mamba_out_rain = self._mamba_forward(rainfall_clustered, self.mamba_input_proj_rain)
        
        # 3. Combine and Predict
        combined = torch.cat([mamba_out_non_rain, mamba_out_rain], dim=1) # (batch_size, 2 * d_model)
        
        # Final prediction
        output = self.combined_head(combined)
        return output.squeeze(-1)