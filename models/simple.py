import torch
import torch.nn as nn
import globals 
from eval import ComboLoss, ThreatScoreLogitsLoss, WeightedComboLoss

class ResBlock(nn.Module):
    def __init__(self, input_dim, d_model, dropout):
        super(ResBlock, self).__init__()

        self.channel = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.channel(x)

        return x

class SimpleModel(nn.Module):
    def __init__(self, features, hyper):
        super(SimpleModel, self).__init__()
        self.name = "SimpleModel"
        self.hyperparameters = hyper
        layers = hyper['layers']
        d_model = hyper['d_model']
        dropout = hyper['dropout']

        self.feature_indices = [globals.all_features.index(feat) for feat in features if feat in globals.all_features]
        self.layer = layers
        num_features = len(self.feature_indices)
        input_dim = num_features * 2

        self.missing_embedding = nn.Parameter(torch.zeros(num_features))
        self.model = nn.ModuleList([ResBlock(input_dim=input_dim, d_model=d_model, dropout=dropout) for _ in range(layers)])
        
        #self.projection = nn.Linear(input_dim, 1)
        bottleneck_dim = 8 # Choose a size smaller than input_dim (F)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim), # Step 1: Compress features
            nn.ReLU(),
            nn.Dropout(dropout), # Apply dropout here to regularize the final decision
            nn.Linear(bottleneck_dim, 1) # Step 2: Final projection to 1
        )

        self.loss_fn = ComboLoss()
        self.missing_value=-1
        #self.loss_fn = nn.BCEWithLogitsLoss()
        #self.loss_fn = ThreatScoreLogitsLoss()

    def forward(self, x, target=None):
        x = x[:, 0, self.feature_indices]
        
        missing_mask = (x == self.missing_value)
        x_filled = torch.where(missing_mask, self.missing_embedding.unsqueeze(0), x)
        missing_indicator = missing_mask.float()

        x = torch.cat([x_filled, missing_indicator], dim=-1)

        for block in self.model:
            x = block(x)

        output = self.projection(x).squeeze(-1)
        probs = torch.sigmoid(output)

        if target is not None:
            loss = self.loss_fn(output, target)
            return probs, loss

        return probs, None

class SimpleAttention(nn.Module):
    def __init__(self, features, hyper):
        super(SimpleAttention, self).__init__()
        self.name = "SimpleAttention"
        self.hyperparameters = hyper
        layers = hyper['layers']
        d_model = hyper['d_model']
        dropout = hyper['dropout']

        self.feature_indices = [globals.all_features.index(feat) for feat in features if feat in globals.all_features]
        self.layer = layers
        num_features = len(self.feature_indices)
        feature_dim = 16
        input_dim = num_features * feature_dim

        self.feature_encoding = nn.Parameter(torch.zeros(num_features, feature_dim))
        self.feature_projection = nn.Linear(2, feature_dim)
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=4, dropout=dropout, batch_first=True)

        self.missing_embedding = nn.Parameter(torch.zeros(num_features))
        self.model = nn.ModuleList([ResBlock(input_dim=input_dim, d_model=d_model, dropout=dropout) for _ in range(layers)])
        
        #self.projection = nn.Linear(input_dim, 1)
        bottleneck_dim = 16 # Choose a size smaller than input_dim (F)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim), # Step 1: Compress features
            nn.ReLU(),
            nn.Dropout(dropout), # Apply dropout here to regularize the final decision
            nn.Linear(bottleneck_dim, 1) # Step 2: Final projection to 1
        )

        self.loss_fn = ComboLoss()
        self.missing_value=-1
        #self.loss_fn = nn.BCEWithLogitsLoss()
        #self.loss_fn = ThreatScoreLogitsLoss()

    def forward(self, x, target=None):
        batch_size = x.shape[0]
        x = x[:, 0, self.feature_indices]
        
        missing_mask = (x == self.missing_value)
        x_filled = torch.where(missing_mask, self.missing_embedding.unsqueeze(0), x)
        x_with_indicator = torch.stack([x_filled, missing_mask.float()], dim=-1)  # [batch, num_features, 2]
        
        x_projected = self.feature_projection(x_with_indicator)  # [batch, num_features, feature_dim]
        x_encoded = x_projected + self.feature_encoding.unsqueeze(0)
        x_attended, _ = self.attention(x_encoded, x_encoded, x_encoded)
        x = x_attended.reshape(batch_size, -1)

        for block in self.model:
            x = block(x)

        output = self.projection(x).squeeze(-1)
        probs = torch.sigmoid(output)

        if target is not None:
            loss = self.loss_fn(output, target)
            return probs, loss

        return probs, None

class SimpleMambaBlock(nn.Module):
    """
    A simplified SSM/Mamba-inspired block for sequence processing (treating features as a sequence).
    Applies 1D convolution for sequence interaction and a simple gating mechanism.
    Input/Output shape: [B, L, D] (Batch, Sequence Length [Features], Dimension)
    """
    def __init__(self, d_ssm, dropout, kernel_size=3):
        super().__init__()
        
        # 1. Sequence Interaction (like Mamba's convolution)
        # 1D Conv operates over the sequence length L (num_features)
        self.conv = nn.Conv1d(
            in_channels=d_ssm, 
            out_channels=d_ssm, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2 # Ensure output sequence length is the same
        )
        self.norm = nn.LayerNorm(d_ssm)
        self.act = nn.SiLU()

        # 2. Simplistic Gating/Selectivity
        # Project x into two branches (one for transformation, one for gate)
        self.proj_in = nn.Linear(d_ssm, d_ssm * 2)
        self.proj_out = nn.Linear(d_ssm, d_ssm)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # x is [B, L, D]
        identity = x
        
        # 1. Expand and Split (Mimicking a selective projection)
        # [B, L, D*2] -> [B, L, D] (a) and [B, L, D] (b)
        ab = self.proj_in(x)
        a, b = ab.chunk(2, dim=-1)
        
        # 2. Sequence Interaction (Conv on feature dimension)
        # Transpose for Conv1d: [B, D, L]
        a = a.transpose(1, 2)
        a = self.conv(a)
        a = self.act(a)
        # Transpose back: [B, L, D]
        a = a.transpose(1, 2)
        
        # 3. Gating (Element-wise multiplication)
        # The 'b' branch acts as a selective gate on the 'a' sequence-transformed branch
        x = a * self.act(b)
        
        # 4. Output Projection and Residual
        x = self.proj_out(x)
        x = self.dropout(x)
        x = self.norm(x + identity)

        return x

class SimpleMamba(nn.Module):
    """
    Mamba-inspired model that treats the selected features as a sequence (L=num_features).
    It processes the feature sequence using SimpleMambaBlocks (SSM).
    """
    def __init__(self, features, hyper):
        super(SimpleMamba, self).__init__()
        self.name = "SimpleMamba"
        self.hyperparameters = hyper
        layers = hyper['layers']
        d_model = hyper['d_model'] # This becomes the dimension D of the SSM
        dropout = hyper['dropout']

        self.feature_indices = [globals.all_features.index(feat) for feat in features if feat in globals.all_features]
        self.layer = layers
        num_features = len(self.feature_indices)
        d_ssm = d_model

        self.missing_value = -1

        # 1. Input Preparation
        self.missing_embedding = nn.Parameter(torch.zeros(num_features))
        # Project (value, missing_flag) [2] to the SSM dimension [d_ssm]
        self.feature_projection = nn.Linear(2, d_ssm) 
        # Feature encoding parameter (similar to positional encoding)
        self.feature_encoding = nn.Parameter(torch.randn(1, num_features, d_ssm) * 0.02) 

        # 2. Core SSM Blocks
        self.model = nn.ModuleList([
            SimpleMambaBlock(d_ssm=d_ssm, dropout=dropout) for _ in range(layers)
        ])

        # 3. Aggregation & Projection
        # After SSM, we have [B, L, D]. Flatten all features (L*D) and project.
        agg_dim = num_features * d_ssm
        bottleneck_dim = d_ssm # Use d_ssm as the bottleneck size
        self.projection = nn.Sequential(
            nn.Linear(agg_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1)
        )
        
        self.loss_fn = ComboLoss()

    def forward(self, x, target=None):
        batch_size = x.shape[0]
        # Select features from the single time step
        x_raw = x[:, 0, self.feature_indices]
        
        missing_mask = (x_raw == self.missing_value)
        x_filled = torch.where(missing_mask, self.missing_embedding.unsqueeze(0), x_raw)
        
        # Stack value and missing indicator: [B, L, 2]
        x_with_indicator = torch.stack([x_filled, missing_mask.float()], dim=-1)

        # 1. Project and encode
        x = self.feature_projection(x_with_indicator) # [B, L, D]
        x = x + self.feature_encoding # Apply feature encoding

        # 2. SSM Layers
        for block in self.model:
            x = block(x)

        # 3. Flatten features: [B, L, D] -> [B, L*D]
        x = x.reshape(batch_size, -1)

        # 4. Final projection
        output = self.projection(x).squeeze(-1)
        probs = torch.sigmoid(output)

        if target is not None:
            loss = self.loss_fn(output, target)
            return probs, loss

        return probs, None

class FeatureGroupModel(nn.Module):
    def __init__(self, pos_weight, d_model=32, dropout=0.15):
        super().__init__()
        self.name = "FeatureGroupModel"
        self.missing_value = -1
        self.hyperparameters = {}

        # Define feature groups
        self.groups = {
            'intensity': ['Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h', 'StormAvgI_mm/h'],
            'accumulation': ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm'],
            'burn': ['dNBR/1000', 'PropHM23'],
            'soil': ['KF'],
            'spatial': ['Latitude', 'Longitude', 'GaugeDist_m', 'ContributingArea_km2'],
            'temporal': ['StormDur_H', 'StormMonth'],
            'interactions': ['Burn_x_Peak', 'KF_x_I15', 'Storm_Energy', 'Erosion_Potential']
        }
        
        # Create indices for each group
        self.group_indices = {}
        for group_name, features in self.groups.items():
            self.group_indices[group_name] = [
                globals.all_features.index(f) for f in features if f in globals.all_features
            ]
        
        # Process each group separately
        self.group_processors = nn.ModuleDict()
        group_output_dim = 16
        
        for group_name, indices in self.group_indices.items():
            if len(indices) > 0:
                self.group_processors[group_name] = nn.Sequential(
                    nn.Linear(len(indices) * 2, group_output_dim),  # *2 for missing indicators
                    nn.GELU(),
                    nn.LayerNorm(group_output_dim),
                    nn.Dropout(dropout)
                )
        
        # Combine groups
        total_dim = len(self.group_processors) * group_output_dim
        
        self.combiner = nn.Sequential(
            nn.Linear(total_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1)
        )
        
        self.loss_fn = WeightedComboLoss(pos_weight=pos_weight)
    
    def forward(self, x, target=None):
        batch_size = x.shape[0]
        x_full = x[:, 0, :]
        
        # Process each group
        group_outputs = []
        for group_name, indices in self.group_indices.items():
            if len(indices) == 0:
                continue
                
            # Extract group features
            group_x = x_full[:, indices]
            
            # Handle missing values
            missing_mask = (group_x == self.missing_value).float()
            group_x = torch.where(group_x == self.missing_value, 
                                 torch.zeros_like(group_x), group_x)
            
            # Concatenate features with missing indicators
            group_input = torch.cat([group_x, missing_mask], dim=-1)
            
            # Process group
            group_output = self.group_processors[group_name](group_input)
            group_outputs.append(group_output)
        
        # Combine all groups
        combined = torch.cat(group_outputs, dim=-1)
        x = self.combiner(combined)
        
        # Final prediction
        output = self.projection(x).squeeze(-1)
        probs = torch.sigmoid(output)
        
        if target is not None:
            loss = self.loss_fn(output, target)
            return probs, loss
        return probs, None