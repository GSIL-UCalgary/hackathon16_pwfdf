import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from models.mamba import threat_score_loss
import data

class GraphMambaModel(nn.Module):
    """
    Hybrid Graph Mamba model combining feature grouping with spatial/spectral branches.
    
    Architecture:
    1. Groups features into semantic pathways (Fire, Terrain, Rain, etc.)
    2. Processes each group through specialized pathways
    3. Splits processing into:
       - Spectral branch: central node features
       - Spatial branch: neighborhood aggregation
    4. Fuses both branches for final prediction
    """
    def __init__(self, max_neighbors, hidden_dim=64, d_model=64, n_mamba_layers=2, dropout=0.1):
        super().__init__()
        self.name = 'GraphMambaMultiPathway'
        
        self.K = max_neighbors + 1
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.dropout_p = dropout
        
        # --- 1. Define Feature Groups (from MultiPathwayHybrid) ---
        self.feature_groups = {
            'Fire': ['Fire_ID', 'Fire_SegID'],
            'Terrain': ['PropHM23', 'ContributingArea_km2'],
            'Burn': ['dNBR/1000', 'PropHM23'],
            'Soil': ['KF', 'KF_Acc015'],
            'Rain_Accumulation': ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm'],
            'Rain_Intensity': ['Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h', 'StormAvgI_mm/h'],
            'Storm': ['StormDur_H', 'GaugeDist_m']
        }
        
        # Determine feature indices for each group
        self.group_indices = {}
        self.pathway_modules = nn.ModuleDict()
        
        for group_name, group_list in self.feature_groups.items():
            indices = [data.all_features.index(feat) for feat in group_list if feat in data.all_features]
            self.group_indices[group_name] = indices
            input_dim = len(indices)
            
            # Define specialized pathways for each group
            if group_name in ['Fire', 'Terrain', 'Soil', 'Storm']:
                # Simple MLP for static features
                self.pathway_modules[group_name] = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, input_dim * 2),
                    nn.ReLU(),
                    nn.Linear(input_dim * 2, 8)  # Output: 8 features per group
                )
                
            elif group_name in ['Burn']:
                # Dedicated small pathway
                self.pathway_modules[group_name] = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, 8)  # Output: 8 features
                )
                
            elif group_name in ['Rain_Accumulation', 'Rain_Intensity']:
                # Mamba pathway for complex features
                self.pathway_modules[group_name] = nn.ModuleDict({
                    'norm': nn.LayerNorm(input_dim),
                    'proj': nn.Linear(input_dim, d_model),
                    'mamba': Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                    'out_norm': nn.LayerNorm(d_model)
                })
        
        # Calculate total pathway output dimension
        self.output_dim_map = {
            'Fire': 8, 'Terrain': 8, 'Burn': 8, 'Soil': 8, 
            'Rain_Accumulation': d_model, 'Rain_Intensity': d_model, 'Storm': 8
        }
        pathway_output_dim = sum(self.output_dim_map.values())
        
        # Project combined pathway outputs to hidden_dim
        self.pathway_fusion = nn.Sequential(
            nn.Linear(pathway_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # --- 2. Spatial Branch (Processes K nodes as sequence with Mamba) ---
        # Only spatial branch uses Mamba for neighborhood aggregation
        self.spatial_mamba1 = Mamba(hidden_dim, hidden_dim * 2)
        self.spatial_norm1 = nn.LayerNorm(hidden_dim)
        
        self.spatial_mamba2 = Mamba(hidden_dim, hidden_dim * 2)
        self.spatial_norm2 = nn.LayerNorm(hidden_dim)
        
        self.spatial_pool = nn.AdaptiveAvgPool1d(1)
        
        # --- 3. Spectral Branch (Processes central node with simple MLP) ---
        self.spectral_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # --- 4. Final Fusion and Classification ---
        fused_dim = hidden_dim + hidden_dim  # spectral + spatial
        
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()

        self.criterion = nn.BCELoss()

    def _init_weights(self):
        """Initialize weights to prevent gradient issues"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _mamba_forward(self, x, pathway_modules):
        """Forward pass for Mamba-based pathways"""
        # x shape: (batch_size, seq_len, feature_dim) or (batch_size, feature_dim)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension: (B, 1, F)
        
        x = pathway_modules['norm'](x)
        x = pathway_modules['proj'](x)  # (B, seq_len, d_model)
        
        residual = x
        x = pathway_modules['mamba'](x)
        x = pathway_modules['out_norm'](x)
        x = residual + x  # Residual connection
        
        x = x.mean(dim=1)  # Pool over sequence: (B, d_model)
        return x

    def _process_pathways(self, x):
        """
        Process input through feature-group pathways.
        
        Args:
            x: (B, K, F) or (B, F) tensor
        
        Returns:
            (B, K, pathway_output_dim) or (B, pathway_output_dim) tensor
        """
        is_spatial = (x.dim() == 3)  # True if (B, K, F), False if (B, F)
        
        if is_spatial:
            B, K, F = x.shape
            # Process each position in K separately
            outputs = []
            for k in range(K):
                x_k = x[:, k, :]  # (B, F)
                k_outputs = []
                
                for group_name, indices in self.group_indices.items():
                    group_x = x_k[:, indices]
                    pathway_modules = self.pathway_modules[group_name]
                    
                    if group_name in ['Rain_Accumulation', 'Rain_Intensity']:
                        out = self._mamba_forward(group_x, pathway_modules)
                    else:
                        out = pathway_modules(group_x)
                    
                    k_outputs.append(out)
                
                combined_k = torch.cat(k_outputs, dim=1)  # (B, pathway_output_dim)
                outputs.append(combined_k)
            
            # Stack all K positions: (B, K, pathway_output_dim)
            return torch.stack(outputs, dim=1)
        
        else:
            # Process single node (B, F)
            outputs = []
            for group_name, indices in self.group_indices.items():
                group_x = x[:, indices]
                pathway_modules = self.pathway_modules[group_name]
                
                if group_name in ['Rain_Accumulation', 'Rain_Intensity']:
                    out = self._mamba_forward(group_x, pathway_modules)
                else:
                    out = pathway_modules(group_x)
                
                outputs.append(out)
            
            return torch.cat(outputs, dim=1)  # (B, pathway_output_dim)

    def forward(self, x, target=None):
        """
        Args:
            x (Tensor): Input tensor of shape (B, K, F)
        """
        B, K, F = x.shape
        
        # --- 1. Process through feature-group pathways ---
        # Process all K nodes through pathways
        pathway_features = self._process_pathways(x)  # (B, K, pathway_output_dim)
        
        # Project to hidden dimension
        x_proj = self.pathway_fusion(pathway_features)  # (B, K, hidden_dim)
        
        # --- 2. Spectral Branch (Central Node with MLP) ---
        x_spec = x_proj[:, 0, :]  # (B, hidden_dim) - just the central node
        
        # Simple MLP processing (rain features already processed by Mamba in pathways)
        spectral_out = self.spectral_mlp(x_spec)  # (B, hidden_dim)
        
        # --- 3. Spatial Branch (All K Nodes) ---
        x_spatial = x_proj  # (B, K, hidden_dim)
        
        # Layer 1
        residual = x_spatial
        spatial_out = self.spatial_mamba1(x_spatial)
        spatial_out = self.spatial_norm1(spatial_out)
        spatial_out = self.dropout_layer(spatial_out)
        spatial_out = residual + spatial_out
        
        # Layer 2
        residual = spatial_out
        spatial_out = self.spatial_mamba2(spatial_out)
        spatial_out = self.spatial_norm2(spatial_out)
        spatial_out = self.dropout_layer(spatial_out)
        spatial_out = residual + spatial_out
        
        # Pool over K dimension
        spatial_pooled = spatial_out.transpose(1, 2)  # (B, hidden_dim, K)
        spatial_pooled = self.spatial_pool(spatial_pooled).squeeze(-1)  # (B, hidden_dim)
        
        # --- 4. Fusion and Classification ---
        fused_features = torch.cat([spectral_out, spatial_pooled], dim=1)  # (B, 2*hidden_dim)
        
        logits = self.classifier(fused_features).squeeze(1)
        
        # Loss calculation
        if target is not None:
            #loss = threat_score_loss(logits, target)
            loss = self.criterion(torch.sigmoid(logits), target)
        else:
            loss = None
        
        return torch.sigmoid(logits), loss
    
class KNNMambaClassifier(nn.Module):
    # New signature: Takes both the full feature list and the limited list
    def __init__(self, all_features, pos_weight, d_model=64, n_layers=2):
        super().__init__()
        self.name = 'KNNMambaClassifier '
        limited_features = ['Acc015_mm', 'Peak_I15_mm/h', 'PropHM23', 'dNBR/1000'] # F=4

        # Store all features for reference
        self.all_features = all_features
        
        # --- Feature Selection Setup ---
        # Calculate the indices of the limited features in the input tensor (F_all)
        self.feature_indices = [all_features.index(f) for f in limited_features]
        # The true input dimension for the projection layer is the length of the limited list
        self.input_dim = len(limited_features) 
        
        self.d_model = d_model
        self.pos_weight = pos_weight
        
        # 1. Feature Projection (F_limited -> D)
        # It now projects the dimension F_limited (e.g., 4) to d_model (64)
        self.projection = nn.Linear(self.input_dim, d_model)
        
        # 2. Mamba Block (N=6 sequence length)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4)
            for _ in range(n_layers)
        ])
        
        # 3. Normalization and Dropout
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(0.1)

        # 4. Classification Head (D -> 1)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.position_embedding = nn.Embedding(6, d_model)

    def forward(self, x, target=None):
        # Input x shape: (B, N=6, F_all)
        
        # --- FEATURE SELECTION STEP ---
        # Selects the limited features (F_limited) using pre-calculated indices
        # x_selected shape: (B, 6, F_limited)
        x_selected = x[:, :, self.feature_indices]
        
        B, N, F_limited = x_selected.shape
        
        # 1. Project features: (B, 6, F_limited) -> (B, 6, D)
        x_proj = self.projection(x_selected)

        # 2. Mamba Layers
        for mamba_layer, norm in zip(self.mamba_layers, self.norms):
            residual = x_proj
            x_proj = mamba_layer(x_proj)
            x_proj = norm(x_proj)
            x_proj = self.dropout(x_proj)
            x_proj = residual + x_proj

        primary_output = x_proj[:, 0, :]   
        neighbor_mean = x_proj[:, 1:, :].mean(dim=1)  # (B, D)
        neighbor_max = x_proj[:, 1:, :].max(dim=1)[0]  # (B, D) 

        target_node_output = torch.cat([primary_output, neighbor_mean, neighbor_max], dim=-1)  # (B, 3*D)

        output = self.classification_head(target_node_output).squeeze(-1)

        loss = None
        if target is not None:
            # Loss calculation using pos_weight
            pos_weight_tensor = torch.tensor([self.pos_weight]).to(output.device)
            loss = F.binary_cross_entropy(output, target, weight=target * (pos_weight_tensor - 1) + 1, reduction='mean')

        return output, loss