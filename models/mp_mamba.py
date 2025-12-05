import torch
import torch.nn as nn
from mamba_ssm import Mamba 

from models.mamba import threat_score_loss
from eval import ThreatScoreLoss, ComboLoss
import globals

class MultiPathwayHybridModel_og(nn.Module):
    def __init__(self, features, d_model=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.duration = '15min'
        self.name = 'MultiPathwayHybrid'
        self.spatial = False
            
        self.hyperparameters = {
            'd_model': d_model,
            #'d_state': d_state,
            #'d_conv': d_conv,
            #'expand': expand,
            'n_layers': n_layers,
            'dropout': dropout
        }

        # --- 1. Define Feature Groups ---
        self.feature_groups_old = {
            #'Fire': ['Fire_ID', 'Fire_SegID'],
            'Terrain': ['PropHM23', 'ContributingArea_km2'],
            'Burn': ['dNBR/1000', 'PropHM23'],
            'Soil': ['KF'],
            'Rain_Accumulation': ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm'],
            'Rain_Intensity': ['Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h', 'StormAvgI_mm/h'],
            'Storm': ['StormDur_H', 'GaugeDist_m']
        }

        self.feature_groups = {
            #'Fire': ['Fire_ID', 'Fire_SegID'],
            'Fire': ['Latitude', 'Longitude'],
            'Terrain': ['PropHM23', 'ContributingArea_km2', 'Area_x_Distance', 'Area_per_Distance'],
            'Burn': ['dNBR/1000', 'PropHM23', 'Burn_x_Peak', 'Burn_x_Accum', 'Burn_x_KF', 
                    'HighMod_x_Peak', 'High_Burn_Severity'],
            'Soil': ['KF', 'KF_x_I15', 'KF_x_I30', 'KF_x_Accum', 'Erosion_Potential'],
            'Rain_Accumulation': ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm',
                                'Early_to_Total_Ratio', 'Early_to_Late_Ratio'],
            'Rain_Intensity': ['Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h', 'StormAvgI_mm/h',
                            'I15_to_Duration', 'I30_to_Duration', 'Peak_to_Avg_Ratio',
                            'Exceeds_I15_Threshold', 'Exceeds_I30_Threshold', 'Storm_Energy'],
            'Storm': ['StormDur_H', 'GaugeDist_m', 'StormMonth', 'Is_Summer', 'Is_Monsoon']
        }
        
        # --- 2. Determine Feature Indices and Pathway Dimensions ---
        self.group_indices = {}
        self.pathway_modules = nn.ModuleDict()

        for group_name, group_list in self.feature_groups.items():
            indices = [globals.all_features.index(feat) for feat in group_list if feat in features]
            self.group_indices[group_name] = indices
            input_dim = len(indices)
            
            print(f"{group_name}: {input_dim}")

            # Define specialized pathways for each group
            if group_name in ['Fire', 'Terrain', 'Soil', 'Storm']:
                # Simple MLP/Linear layer for static/simple features
                self.pathway_modules[group_name] = nn.Sequential(
                    nn.Linear(input_dim, input_dim * 2),
                    nn.ReLU(),
                    nn.Linear(input_dim * 2, 4) # Output feature size is 8
                )
                
            elif group_name in ['Burn']:
                # Dedicated small pathway
                self.pathway_modules[group_name] = nn.Sequential(
                    nn.Linear(input_dim, 4), # Output feature size is 4
                )
                
            elif group_name in ['Rain_Accumulation', 'Rain_Intensity']:
                # Mamba pathway for sequence-like/complex time-series features
                # The Mamba layer will take the input dim and project to d_model, then process
                mamba_input_proj = nn.Linear(1, d_model)
                mamba_layers = nn.ModuleList([
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2,) 
                    for _ in range(n_layers)
                ])
                mamba_norms = nn.ModuleList([
                    nn.LayerNorm(d_model)
                    for _ in range(n_layers)
                ])
                
                # Store the components in a ModuleDict for easy access
                self.pathway_modules[group_name] = nn.ModuleDict({
                    'proj': mamba_input_proj,
                    'layers': mamba_layers,
                    'norms': mamba_norms
                })
        
        # Calculate the total concatenated dimension for the combined head
        # Fire (8) + Terrain (8) + Burn (4) + Soil (8) + Rain_Accumulation (d_model=64) + Rain_Intensity (d_model=64) + Storm (8)
        self.output_dim_map = {
            'Fire': 4, 'Terrain': 4, 'Burn': 4, 'Soil': 4, 
            'Rain_Accumulation': d_model, 'Rain_Intensity': d_model, 'Storm': 4
        }
        total_combined_dim = sum(self.output_dim_map.values())
        
        # --- 3. Combined Output Head ---
        self.combined_head = nn.Sequential(
            nn.Linear(total_combined_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            #nn.Sigmoid() #for binary classification if needed
        )
        
        self.dropout = nn.Dropout(dropout)
        #self.loss_fn = ThreatScoreLoss()
        self.loss_fn = ComboLoss()
        #self.loss_fn = nn.BCELoss()

    def _mamba_forward(self, x, pathway_modules):
        """Dedicated forward pass for Mamba pathways."""
        
        # x shape: (batch_size, feature_dim)
        x = x.unsqueeze(-1) # (batch_size, 1, feature_dim) - Mamba expects a sequence dimension
        x = pathway_modules['proj'](x) # (batch_size, 1, d_model)
        
        # Mamba layers
        for mamba_layer, norm in zip(pathway_modules['layers'], pathway_modules['norms']):
            residual = x
            x = mamba_layer(x)
            x = norm(x)
            x = self.dropout(x)
            x = residual + x # Residual connection
            
        x = x.mean(dim=1) # (batch_size, d_model) - Global pooling over sequence dimension
        return x

    def forward(self, x, target=None):
        x = x[:, 0, :]
        batch_outputs = []
        
        for group_name, indices in self.group_indices.items():
            # 1. Split features
            group_x = x[:, indices]
            pathway_modules = self.pathway_modules[group_name]
            
            # 2. Process through the specialized pathway
            if group_name in ['Rain_Accumulation', 'Rain_Intensity']:
                # Mamba Pathway
                out = self._mamba_forward(group_x, pathway_modules)
            else:
                # MLP/Linear Pathway
                out = pathway_modules(group_x)
            
            batch_outputs.append(out)
            
        combined = torch.cat(batch_outputs, dim=1)
        output = self.combined_head(combined).squeeze(-1)
        probs = torch.sigmoid(output)

        if target != None:
            #loss = threat_score_loss(output, target)
            loss = self.loss_fn(output, target)
        else:
            loss = None

        return probs, loss
    