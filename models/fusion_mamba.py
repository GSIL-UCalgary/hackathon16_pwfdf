import torch
import torch.nn as nn
from mamba_ssm import Mamba

from eval import ThreatScoreLoss

class FeatureGroupBranch(nn.Module):
    """Single branch for processing one feature group"""
    def __init__(self, n_features, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        
        # Project features to d_model dimension
        self.input_proj = nn.Linear(n_features, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Mamba block
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Optional feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)
        x = self.norm1(x)
        
        # Mamba processing with residual
        x = x + self.mamba(x)
        x = self.norm2(x)
        
        # Feedforward with residual
        x = x + self.ff(x)
        x = self.norm3(x)
        
        return x

class MultiBranchMamba(nn.Module):
    """Mamba model with separate branches for different feature groups"""
    def __init__(
        self,
        feature_names,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        n_classes=1,
        fusion_method='concat',  # 'concat', 'sum', 'attention'
    ):
        super().__init__()

        self.feature_groups = {
            #'Fire': ['Fire_ID', 'Fire_SegID'],
            'Terrain': ['PropHM23', 'ContributingArea_km2'],
            'Burn': ['dNBR/1000', 'PropHM23'],
            'Soil': ['KF', 'KF_Acc015'],
            'Rain_Accumulation': ['Acc015_mm']#, 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm'],
            #'Rain_Intensity': ['Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h', 'StormAvgI_mm/h'],
            #'Storm': ['StormDur_H', 'GaugeDist_m']
        }
        self.name = f"MultiBranchMamba_{fusion_method}"
        self.feature_groups = self.feature_groups
        self.feature_names = feature_names
        self.d_model = d_model
        self.fusion_method = fusion_method
        self.n_classes = n_classes
        
        # Create feature index mapping
        self.feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        # Identify grouped and ungrouped features
        self.grouped_features = set()
        self.group_indices = {}
        
        for group_name, features in self.feature_groups.items():
            indices = []
            for feat in features:
                if feat in self.feature_to_idx:
                    idx = self.feature_to_idx[feat]
                    indices.append(idx)
                    self.grouped_features.add(feat)
            self.group_indices[group_name] = indices
        
        # Ungrouped features
        self.ungrouped_indices = [
            self.feature_to_idx[feat] 
            for feat in feature_names 
            if feat not in self.grouped_features
        ]
        
        # Create branches for each feature group
        self.branches = nn.ModuleDict()
        for group_name, features in self.feature_groups.items():
            if self.group_indices[group_name]:  # Only create if features exist
                self.branches[group_name] = FeatureGroupBranch(
                    n_features=len(self.group_indices[group_name]),
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
        
        # Branch for ungrouped features
        if self.ungrouped_indices:
            self.branches['ungrouped'] = FeatureGroupBranch(
                n_features=len(self.ungrouped_indices),
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        
        # Fusion layer
        n_branches = len(self.branches)
        if fusion_method == 'concat':
            fusion_dim = d_model * n_branches
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model)
            )
        elif fusion_method == 'sum':
            self.fusion = nn.Identity()
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                batch_first=True
            )
            self.fusion = nn.Linear(d_model, d_model)
        
        # Output head
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes),
            nn.Sigmoid()
        )
        
        self.loss_fn = nn.BCELoss()
        #self.loss_fn = ThreatScoreLoss()

    
    def forward(self, x, targets=None):
        """
        Args:
            x: Input tensor of shape (batch, N, F)
               where N=0 is the primary node to predict
            targets: Target tensor of shape (batch,) or (batch, n_classes)
                    Only used during training
        
        Returns:
            If targets provided: (loss, predictions)
            If targets None: predictions only
        """
        # x: (batch, N, F)
        batch_size = x.shape[0]
        
        # Split features into groups
        branch_outputs = []
        
        for group_name, branch in self.branches.items():
            if group_name == 'ungrouped':
                #indices = self.ungrouped_indices
                indices = None
            else:
                indices = self.group_indices[group_name]
            
            if indices:
                # Extract features for this group
                group_features = x[:, :, indices]  # (batch, N, n_features_in_group)
                
                # Process through branch
                out = branch(group_features)  # (batch, N, d_model)
                branch_outputs.append(out)
        
        # Fusion
        if self.fusion_method == 'concat':
            # Concatenate along feature dimension
            fused = torch.cat(branch_outputs, dim=-1)  # (batch, N, d_model*n_branches)
            fused = self.fusion(fused)  # (batch, N, d_model)
        
        elif self.fusion_method == 'sum':
            # Simple summation
            fused = torch.stack(branch_outputs, dim=0).sum(dim=0)  # (batch, N, d_model)
        
        elif self.fusion_method == 'attention':
            # Stack branches and use cross-attention
            stacked = torch.stack(branch_outputs, dim=2)  # (batch, N, n_branches, d_model)
            b, n, g, d = stacked.shape
            stacked = stacked.reshape(b * n, g, d)
            
            attended, _ = self.attention(stacked, stacked, stacked)
            attended = attended.mean(dim=1)  # Average over branches
            fused = attended.reshape(b, n, d)
            fused = self.fusion(fused)
        
        # Output prediction - use only the primary node (N=0)
        primary_node = fused[:, 0, :]  # (batch, d_model)
        probs = self.output(primary_node)  # (batch, n_classes)
        
        # Squeeze if single class output
        if self.n_classes == 1:
            probs = probs.squeeze(-1)  # (batch,)
        
        # Calculate loss if targets provided
        if targets is not None:
            loss = self.loss_fn(probs, targets)
            return probs, loss
        
        return probs, None


# Example usage
if __name__ == "__main__":
    # Define feature groups

    
    # All feature names (including ungrouped ones)
    feature_names = [
        'Fire_ID', 'Fire_SegID', 'PropHM23', 'ContributingArea_km2',
        'dNBR/1000', 'KF', 'KF_Acc015', 'Acc015_mm', 'Acc030_mm',
        'Acc060_mm', 'StormAccum_mm', 'Peak_I15_mm/h', 'Peak_I30_mm/h',
        'Peak_I60_mm/h', 'StormAvgI_mm/h', 'StormDur_H', 'GaugeDist_m',
        'Ungrouped_Feat1', 'Ungrouped_Feat2'  # Example ungrouped features
    ]
    
    # Create model
    model = MultiBranchMamba(
        feature_names=feature_names,
        d_model=64,
        d_state=16,
        n_classes=1,
        fusion_method='concat',
        loss_fn='mse'
    )
    
    # Create dummy input data
    batch_size = 16
    n_nodes = 10  # N=0 is primary node, N=1..9 are neighbors
    n_features = len(feature_names)
    
    x = torch.randn(batch_size, n_nodes, n_features)
    targets = torch.randn(batch_size)  # Regression targets
    
    # Training mode - returns loss and predictions
    loss, predictions = model(x, targets)
    print(f"Training mode:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Predictions shape: {predictions.shape}")  # (16,)
    
    # Inference mode - returns predictions only
    model.eval()
    with torch.no_grad():
        predictions = model(x)
        print(f"\nInference mode:")
        print(f"  Predictions shape: {predictions.shape}")  # (16,)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Show branch-specific info
    print(f"\nBranch information:")
    for name, branch in model.branches.items():
        branch_params = sum(p.numel() for p in branch.parameters())
        if name == 'ungrouped':
            n_feats = len(model.ungrouped_indices)
        else:
            n_feats = len(model.group_indices[name])
        print(f"  {name}: {n_feats} features, {branch_params:,} parameters")