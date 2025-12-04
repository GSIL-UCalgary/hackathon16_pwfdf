import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from eval import threat_score_loss

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
    def __init__(self, features, pos_weight, input_dim=16, d_model=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.duration = '15min'
        self.name = 'HybridMamba'
        self.spatial = False
        
        features_to_remove = ['UTM_X', 'UTM_Y', 'Fire_ID', 'Fire_SegID']
        self.features = [f for f in features if f not in features_to_remove]
        remove_indices = [features.index(f) for f in features_to_remove]
        #self.utm_x_idx = features.index("UTM_X")
        #self.utm_y_idx = features.index("UTM_Y")

        self.keep_indices = torch.tensor([
            i for i in range(len(features)) 
            #if i not in [self.utm_x_idx, self.utm_y_idx]
            if i not in remove_indices
        ])

        self.rainfall_features = ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm']
        self.all_features = self.features
        
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
            nn.Sigmoid()
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
        #self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))


    def split_features(self, x):
        """Split input into rainfall and non-rainfall features"""
        # x shape: (batch_size, input_dim)
        non_rainfall_x = x[:, self.non_rainfall_indices]  # (batch_size, non_rainfall_dim)
        rainfall_x = x[:, self.rainfall_indices]          # (batch_size, rainfall_dim)
        return non_rainfall_x, rainfall_x
    
    def forward(self, x, target=None):
        x = x[:, self.keep_indices]  # [B, F-2]

        non_rainfall_x, rainfall_x = self.split_features(x)
        
        mamba_out = self._mamba_forward(non_rainfall_x)
        logistic_out = self.logistic_layer(rainfall_x)  # (batch_size, 1)
        
        combined = torch.cat([mamba_out, logistic_out], dim=1)  # (batch_size, d_model + 1)
        
        # Final prediction
        output = self.combined_head(combined).squeeze(-1)

        if target != None:
            #loss = self.criterion(output, target)
            loss = threat_score_loss(output, target)
        else:
            loss = None

        return torch.sigmoid(output), loss
    
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

class HybridMambaFeatureGroups(nn.Module):
    def __init__(self, features, pos_weight=3.0, dropout=0.1):
        super().__init__()
        self.features = features
        self.pos_weight = pos_weight
        self.name = 'HybridMambaFG'
        self.spatial = False
        
        # Define feature groups manually (example)
        self.rainfall_features = ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm']
        self.burn_severity_features = ['PropHM23', 'dNBR/1000', 'KF']
        self.other_features = [f for f in features if f not in self.rainfall_features + self.burn_severity_features]
        
        # Get indices for each group
        self.rainfall_indices = [features.index(f) for f in self.rainfall_features if f in features]
        self.burn_severity_indices = [features.index(f) for f in self.burn_severity_features if f in features]
        self.other_indices = [features.index(f) for f in self.other_features]

        # Submodules per group
        self.rainfall_net = nn.Sequential(
            nn.Linear(len(self.rainfall_indices), 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.burn_severity_net = nn.Sequential(
            nn.Linear(len(self.burn_severity_indices), 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.other_net = nn.Sequential(
            nn.Linear(len(self.other_indices), 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combine outputs from all groups
        self.combined_head = nn.Sequential(
            nn.Linear(16 * 3, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, target=None):
        # Extract each group
        rainfall_x = x[:, self.rainfall_indices]
        burn_x = x[:, self.burn_severity_indices]
        other_x = x[:, self.other_indices]
        
        # Forward pass through each group net
        rainfall_out = self.rainfall_net(rainfall_x)
        burn_out = self.burn_severity_net(burn_x)
        other_out = self.other_net(other_x)
        
        # Concatenate group outputs
        combined = torch.cat([rainfall_out, burn_out, other_out], dim=1)
        
        # Final output
        logits = self.combined_head(combined).squeeze(-1)
        
        if target is not None:
            loss = self.combined_loss(logits, target)
        else:
            loss = None
        
        return torch.sigmoid(logits), loss
    
    def combined_loss(self, preds, targets, alpha=0.5, gamma=2.0):
        """
        Combine BCE with logits (with pos_weight) and threat score loss.
        """
        # BCE loss with positive class weighting
        pos_weight = self.pos_weight
        if not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        pos_weight = pos_weight.to(preds.device)
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=pos_weight)    

        # Threat score loss: maximize intersection over union
        preds_prob = torch.sigmoid(preds)
        preds_bin = (preds_prob > 0.5).float()
        targets = targets.float()
        
        intersection = (preds_bin * targets).sum()
        union = preds_bin.sum() + targets.sum() - intersection + 1e-7
        ts_loss = 1.0 - (intersection / union)
        
        # Optionally add focal loss component
        pt = torch.where(targets == 1, preds_prob, 1 - preds_prob)
        focal_weight = (1 - pt) ** gamma
        focal_loss = (focal_weight * F.binary_cross_entropy_with_logits(preds, targets, reduction='none')).mean()
        
        # Weighted sum of losses
        loss = alpha * bce_loss + (1 - alpha) * ts_loss  # + 0.1 * focal_loss  (optional)
        return loss

class HybridMambaLogisticModelPro(nn.Module):
    def __init__(self, features, pos_weight, d_model=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.name = 'HybridMambaPro'

        # --- Feature Indexing (Unchanged) ---
        self.rainfall_features = ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm']
        self.all_features = features
        self.rainfall_indices = [features.index(feat) for feat in self.rainfall_features if feat in features]
        self.non_rainfall_indices = [i for i in range(len(features)) if i not in self.rainfall_indices]
        
        # --- Mamba Pathway ---
        self.mamba_input_dim = len(self.non_rainfall_indices)
        self.mamba_input_proj = nn.Sequential(
            nn.Linear(self.mamba_input_dim, d_model),
            nn.ReLU(),
            self.dropout
        )
        
        # Increased Mamba Capacity: expand=3
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=32, d_conv=4, expand=3,) # Increased d_state and expand
            for _ in range(n_layers)
        ])
        
        # Layer normalization for Mamba
        self.mamba_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        # --- Logistic Pathway ---
        self.logistic_input_dim = len(self.rainfall_indices)
        self.logistic_layer = nn.Sequential(
            nn.Linear(self.logistic_input_dim, 1),
            nn.Sigmoid() 
        )
        
        # --- Gated Fusion and Output Head (Replaced Combined Head) ---
        # 1. Gate projection: Maps the single logistic output (B, 1) to the d_model size (B, d_model)
        self.gate_proj = nn.Linear(1, d_model)
        
        # 2. Final prediction head (Input is d_model after gating)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            self.dropout,
            nn.Linear(32, 1), # Output logits
        )
        
        # Loss function place holder (Threat Score Loss is handled externally)
        self.bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def split_features(self, x):
        """Split input into rainfall and non-rainfall features"""
        non_rainfall_x = x[:, self.non_rainfall_indices] 
        rainfall_x = x[:, self.rainfall_indices] 
        return non_rainfall_x, rainfall_x
    
    def forward(self, x, target=None):
        non_rainfall_x, rainfall_x = self.split_features(x)
        
        # --- Pathway Outputs ---
        mamba_out = self._mamba_forward(non_rainfall_x)  # (B, d_model)
        logistic_out = self.logistic_layer(rainfall_x)   # (B, 1)
        
        # --- Gated Fusion ---
        # 1. Project logistic output to d_model size
        gate_signal = self.gate_proj(logistic_out)      # (B, d_model)
        
        # 2. Gating: Use the projected signal to scale the Mamba output
        x_fused = mamba_out * gate_signal              # (B, d_model)
        
        # 3. Final Prediction (Logits)
        output = self.output_head(x_fused).squeeze(-1) # Output shape (B,)

        if target is not None:
            # Calculate Hybrid Loss using external threat_score_loss and internal BCE
            loss = self.calculate_hybrid_loss(output, target)
        else:
            loss = None

        return torch.sigmoid(output), loss
    
    def _mamba_forward(self, x):
        # 1. Add Sequence Dimension and Project
        x = x.unsqueeze(1)  # (batch_size, L=1, mamba_input_dim)
        x = self.mamba_input_proj(x)  # (batch_size, 1, d_model)
        
        # 2. Mamba Backbone (Pre-Norm style)
        for mamba_layer, norm in zip(self.mamba_layers, self.mamba_norms):
            residual = x
            
            # Pre-Norm: Normalize before feeding into the block
            x = norm(x)
            x = mamba_layer(x)
            
            x = self.dropout(x)
            x = residual + x # Add residual after dropout
        
        # 3. Remove Sequence Dimension
        x = x.squeeze(dim=1)  # (batch_size, d_model)
        return x
    
    def calculate_hybrid_loss(self, output_logits, target, lambda_ts=0.5):
        """Calculates a weighted average of Threat Score Loss and BCE Loss."""
        # Ensure target is float for both losses
        target = target.float() 
        
        # L_TS: Maximizing TS is equivalent to minimizing -TS
        # Assuming threat_score_loss is defined globally
        loss_ts = threat_score_loss(output_logits, target) 
        
        # L_BCE: Standard stability loss
        loss_bce = self.bce_criterion(output_logits, target)
        
        # Weighted hybrid loss
        hybrid_loss = lambda_ts * loss_ts + (1 - lambda_ts) * loss_bce
        return hybrid_loss
    

class SpatialMambaHybridModel(nn.Module):
    def __init__(self, features, pos_weight, d_model=64, n_layers=4, dropout=0.1):
        super().__init__()
        self.name = 'SpatialMambaHybrid_vFinal'
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # ------------------------------------------------------------------
        # 1. Feature Engineering & Grouping
        # ------------------------------------------------------------------
        self.features = features
        
        # A. Rainfall Features (The "Trigger")
        self.rain_feats = ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm']
        self.rain_idx = [features.index(f) for f in self.rain_feats if f in features]
        
        # B. Topography/Static Features (e.g., Slope, Elevation, KF)
        # Assuming 'KF' is here. Add others like 'Slope', 'Elevation' if you have them.
        self.topo_feats = ['KF'] 
        self.topo_idx = [features.index(f) for f in self.topo_feats if f in features]
        
        # C. Fuel/Vegetation Features (e.g., dNBR, PropHM23)
        self.fuel_feats = ['dNBR/1000', 'PropHM23']
        self.fuel_idx = [features.index(f) for f in self.fuel_feats if f in features]
        
        # Catch-all for any remaining features to go into Fuel or Topo (optional)
        used_indices = set(self.rain_idx + self.topo_idx + self.fuel_idx)
        remaining = [i for i in range(len(features)) if i not in used_indices]
        self.fuel_idx.extend(remaining) # Assign leftovers to fuel branch
        
        # ------------------------------------------------------------------
        # 2. Parallel Mamba Branches (Spatial Context)
        # ------------------------------------------------------------------
        
        # --- Branch A: Topography ---
        self.topo_proj = nn.Sequential(
            nn.Linear(len(self.topo_idx), d_model),
            nn.ReLU(),
            self.dropout
        )
        self.topo_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=32, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.topo_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # --- Branch B: Fuel/Vegetation ---
        self.fuel_proj = nn.Sequential(
            nn.Linear(len(self.fuel_idx), d_model),
            nn.ReLU(),
            self.dropout
        )
        self.fuel_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=32, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.fuel_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # ------------------------------------------------------------------
        # 3. Logistic Branch (Rainfall Trigger)
        # ------------------------------------------------------------------
        self.logistic_layer = nn.Sequential(
            nn.Linear(len(self.rain_idx), 1),
            nn.Sigmoid() # Output is probability [0, 1]
        )
        
        # ------------------------------------------------------------------
        # 4. Gated Fusion Mechanism
        # ------------------------------------------------------------------
        # Project Logistic output (1 dim) to Gate size (2 * d_model)
        # We want to gate the Combined Mamba output (Topo + Fuel)
        self.gate_proj = nn.Linear(1, 2 * d_model)
        
        # Final prediction head
        self.output_head = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 1) # Final Logits
        )
        
        # Internal Loss Component
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def _process_mamba_branch(self, x_seq, proj, layers, norms):
        """
        Helper to process a sequence through a Mamba stack.
        Input: (Batch, Neighbors, Feats) -> Output: (Batch, d_model)
        """
        # 1. Project to d_model
        x = proj(x_seq) # (B, L, d_model)
        
        # 2. Mamba Blocks (Pre-Norm Residuals)
        for layer, norm in zip(layers, norms):
            residual = x
            x = norm(x)
            x = layer(x)
            x = self.dropout(x)
            x = residual + x
            
        # 3. Temporal Pooling: Average over Neighbors (Sequence Dim 1)
        # This condenses the "Supersample" into one context vector
        x = x.mean(dim=1) 
        return x

    def forward(self, x, target=None):
        # ---------------------------------------------------------
        # Input x shape: (Batch_Size, Neighbors+1, Total_Features)
        # ---------------------------------------------------------
        
        # 1. Slice Features
        # Topo and Fuel use the FULL sequence (Spatial Context)
        x_topo = x[:, :, self.topo_idx] 
        x_fuel = x[:, :, self.fuel_idx]
        
        # Rainfall: We only care about the CENTRAL point (index 0)
        # We assume the "target" point is at index 0 of the neighbor sequence
        x_rain = x[:, 0, self.rain_idx] 
        
        # 2. Run Branches
        topo_vec = self._process_mamba_branch(x_topo, self.topo_proj, self.topo_mamba, self.topo_norms)
        fuel_vec = self._process_mamba_branch(x_fuel, self.fuel_proj, self.fuel_mamba, self.fuel_norms)
        
        rain_prob = self.logistic_layer(x_rain) # (B, 1) range [0,1]
        
        # 3. Gated Fusion
        # Combine Environmental Contexts
        env_context = torch.cat([topo_vec, fuel_vec], dim=-1) # (B, 2*d_model)
        
        # Create Gate from Rainfall Probability
        # If Rain is 0, Gate is near 0 -> Environmental risk doesn't matter
        # If Rain is 1, Gate is near 1 -> Environmental risk is fully active
        gate = self.gate_proj(rain_prob) # (B, 2*d_model)
        
        # Apply Gate (Element-wise multiplication)
        fused_vector = env_context * gate
        
        # 4. Final Prediction
        logits = self.output_head(fused_vector).squeeze(-1) # (B,)
        
        # 5. Loss Calculation
        loss = None
        if target is not None:
            # Hybrid Loss: 70% Threat Score, 30% BCE (Adjust lambda as needed)
            l_ts = self.threat_score_loss(logits, target)
            l_bce = self.bce(logits, target.float())
            loss = 0.7 * l_ts + 0.3 * l_bce
            
        return torch.sigmoid(logits), loss
    
    # --- Helper for the Loss Function ---
    def threat_score_loss(self, outputs, targets, epsilon=1e-6):
        """
        Differentiable Threat Score (IoU) Loss.
        outputs: Logits (before sigmoid)
        targets: Binary labels (0 or 1)
        """
        probs = torch.sigmoid(outputs)
        targets = targets.float()
        
        TP = torch.sum(targets * probs)
        FN = torch.sum(targets * (1 - probs))
        FP = torch.sum((1 - targets) * probs)
        
        threat_score = TP / (TP + FN + FP + epsilon)
        return -threat_score

class SpatialMambaContextModel(nn.Module):
    def __init__(self, features, pos_weight, seq_len, d_model=64, n_layers=6, dropout=0.2):
        super().__init__()
        self.name = 'SpatialMambaContext_vPE'
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len # L = Neighbors + 1 (needed for PE)
        
        # --- Feature Indexing (Identical to previous model) ---
        self.rain_feats = ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm']
        self.topo_feats = ['KF'] # Example
        self.fuel_feats = ['dNBR/1000', 'PropHM23'] # Example
        
        self.rain_idx = [features.index(f) for f in self.rain_feats if f in features]
        self.topo_idx = [features.index(f) for f in self.topo_feats if f in features]
        self.fuel_idx = [features.index(f) for f in self.fuel_feats if f in features]

        # ------------------------------------------------------------------
        # NEW: Context Token and Positional Encoding
        # ------------------------------------------------------------------
        # 1. Learnable Context Token (CLT) - Prepended to the sequence
        self.context_token = nn.Parameter(torch.randn(1, 1, d_model)) 
        
        # 2. Positional Encoding for neighbor rank (L positions)
        self.pos_emb = nn.Embedding(seq_len + 1, d_model) # +1 for the CLT position
        
        # ------------------------------------------------------------------
        # Mamba Branches (Wider State)
        # ------------------------------------------------------------------
        
        # --- Branch A: Topography ---
        self.topo_proj = nn.Sequential(nn.Linear(len(self.topo_idx), d_model), nn.ReLU(), self.dropout)
        self.topo_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=64, d_conv=4, expand=2) # Wider d_state=64
            for _ in range(n_layers)
        ])
        self.topo_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # --- Branch B: Fuel/Vegetation ---
        self.fuel_proj = nn.Sequential(nn.Linear(len(self.fuel_idx), d_model), nn.ReLU(), self.dropout)
        self.fuel_mamba = nn.ModuleList([
            Mamba(d_model=d_model, d_state=64, d_conv=4, expand=2) # Wider d_state=64
            for _ in range(n_layers)
        ])
        self.fuel_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # --- (Logistic and Gating Layers remain the same) ---
        self.logistic_layer = nn.Sequential(nn.Linear(len(self.rain_idx), 1), nn.Sigmoid())
        self.gate_proj = nn.Linear(1, 2 * d_model)
        
        # --- Final Output Head (Input: 2 * d_model) ---
        self.output_head = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 1)
        )
        
        # Internal Loss Component
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def _process_mamba_branch(self, x_seq, proj, layers, norms):
        """Processes sequence with CLT and PE, then extracts CLT for fusion."""
        B, L, F = x_seq.shape
        
        # 1. Feature Projection
        x = proj(x_seq) # (B, L, d_model)
        
        # 2. Add Context Token (CLT)
        clt_token = self.context_token.expand(B, -1, -1) # (B, 1, d_model)
        x = torch.cat([clt_token, x], dim=1) # (B, L+1, d_model)
        
        # 3. Add Positional Encoding (PE)
        positions = torch.arange(L + 1, device=x.device).unsqueeze(0) # (1, L+1)
        x = x + self.pos_emb(positions) # (B, L+1, d_model) + (1, L+1, d_model)

        # 4. Mamba Blocks (Pre-Norm Residuals)
        for layer, norm in zip(layers, norms):
            residual = x
            x = norm(x)
            x = layer(x)
            x = self.dropout(x)
            x = residual + x
            
        # 5. Contextual Pooling: Extract the CLT (index 0)
        x_context = x[:, 0, :] # (B, d_model)
        return x_context

    def forward(self, x, target=None):
        # x shape: (Batch_Size, Neighbors+1, Total_Features)
        
        # 1. Slice Features (Rainfall only uses central point, index 0)
        x_topo = x[:, :, self.topo_idx] 
        x_fuel = x[:, :, self.fuel_idx]
        x_rain = x[:, 0, self.rain_idx] 
        
        # 2. Run Mamba Branches (Spatial Context)
        topo_vec = self._process_mamba_branch(x_topo, self.topo_proj, self.topo_mamba, self.topo_norms)
        fuel_vec = self._process_mamba_branch(x_fuel, self.fuel_proj, self.fuel_mamba, self.fuel_norms)
        
        rain_prob = self.logistic_layer(x_rain) # (B, 1) probability
        
        # 3. Gated Fusion
        env_context = torch.cat([topo_vec, fuel_vec], dim=-1) # (B, 2*d_model)
        gate = self.gate_proj(rain_prob)
        fused_vector = env_context * gate
        
        # 4. Final Prediction
        logits = self.output_head(fused_vector).squeeze(-1) # (B,)
        
        # 5. Loss Calculation
        loss = None
        if target is not None:
            # Aggressive TS optimization (e.g., 0.7/0.3 split)
            l_ts = threat_score_loss(logits, target)
            l_bce = self.bce(logits, target.float())
            loss = 0.7 * l_ts + 0.3 * l_bce
            
        return torch.sigmoid(logits), loss
    
class HybridGroupedSpatialMambaModel(nn.Module):
    """
    Multi-branch model:
      - X_flat        : [B, F]
      - X_spatial     : [B, K, F_spatial]
    
    Each feature group receives its own branch, then outputs are fused.
    """
    def __init__(
        self,
        features,
        spatial_dim,
        d_model=128,
        group_d_model=128,
        spatial_d_model=128,
        dropout=0.1,
        n_layers=4
    ):
        super().__init__()
        self.features = features
        self.dropout = nn.Dropout(dropout)
        self.name = 'HGSMamba'
        self.spatial=True

        self.feature_groups = {
            "rainfall": ["Acc015_mm", "Acc030_mm", "Acc060_mm", "StormAccum_mm"],
            "intensity": ["Peak_I15_mm/h", "Peak_I30_mm/h", "Peak_I60_mm/h"],
            "geomorph": ["ContributingArea_km2", "PropHM23", "dNBR/1000", "KF"],
            "baseline": ["GaugeDist_m", "StormDur_H", "StormAvgI_mm/h"],
        }

        # -----------------------------------------------------------
        # 1. Build index lists for each group
        # -----------------------------------------------------------
        self.group_indices = {}
        for group_name, group_feats in self.feature_groups.items():
            idxs = [features.index(f) for f in group_feats if f in features]
            if len(idxs) == 0:
                raise ValueError(f"Group {group_name} has no valid features.")
            self.group_indices[group_name] = idxs

        # -----------------------------------------------------------
        # 2. Build branches
        # -----------------------------------------------------------
        self.group_branches = nn.ModuleDict()
        for group_name, idxs in self.group_indices.items():
            in_dim = len(idxs)

            # A small MLP branch (can replace with Mamba)
            self.group_branches[group_name] = nn.Sequential(
                nn.Linear(in_dim, group_d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(group_d_model, group_d_model),
                nn.ReLU(),
            )

        # -----------------------------------------------------------
        # 3. Spatial branch (Mamba over k neighbors)
        # -----------------------------------------------------------
        self.spatial_input_proj = nn.Linear(spatial_dim, spatial_d_model)

        from mamba_ssm import Mamba   # assuming your Mamba import

        self.spatial_layers = nn.ModuleList([
            Mamba(d_model=spatial_d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.spatial_norms = nn.ModuleList([nn.LayerNorm(spatial_d_model) for _ in range(n_layers)])

        # -----------------------------------------------------------
        # 4. Combined fusion head
        # -----------------------------------------------------------
        total_dim = len(self.group_branches) * group_d_model + spatial_d_model

        self.fuse_head = nn.Sequential(
            nn.Linear(total_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    # ---------------------------------------------------------------
    # Helper: spatial Mamba forward
    # ---------------------------------------------------------------
    def _spatial_forward(self, x_spatial):
        """
        x_spatial: [B, K, F_spatial]
        """
        x = self.spatial_input_proj(x_spatial)  # [B, K, d]
        for layer, norm in zip(self.spatial_layers, self.spatial_norms):
            residual = x
            x = layer(x)
            x = norm(x)
            x = self.dropout(x)
            x = x + residual

        x = x.mean(dim=1)  # pool over neighbors
        return x

    # ---------------------------------------------------------------
    # Main forward
    # ---------------------------------------------------------------
    def forward(self, x, target=None):
        x_flat, x_spatial = x

        """
        x_flat:    [B, F]
        x_spatial: [B, K, F_spatial]
        """

        group_outputs = []
        for group_name, idxs in self.group_indices.items():
            group_x = x_flat[:, idxs]
            out = self.group_branches[group_name](group_x)
            group_outputs.append(out)

        spatial_out = self._spatial_forward(x_spatial)
        group_outputs.append(spatial_out)

        fused = torch.cat(group_outputs, dim=1)  # [B, total_dim]
        logits = self.fuse_head(fused).squeeze(-1)
        probs = torch.sigmoid(logits)

        if target is not None:
            loss = threat_score_loss(logits, target)
        else:
            loss = None

        return probs, loss

def create_spatial_tensors(X, features, k_neighbors=4):
    """
    X: Tensor [B, F]
    features: list of feature names (strings)
    k_neighbors: number of nearest neighbors

    Returns:
        spatial_X: Tensor [B, k_neighbors, F_without_coords]
    """
    # ----------- 1. Find coordinate indices -----------
    try:
        ix = features.index("UTM_X")
        iy = features.index("UTM_Y")
    except ValueError:
        raise ValueError("Features list must contain 'UTM_X' and 'UTM_Y'.")

    # Coordinates for distance computation
    coords = X[:, [ix, iy]]        # [B, 2]

    # ----------- 2. Compute pairwise UTM distances -----------
    diff = coords[:, None, :] - coords[None, :, :]   # [B, B, 2]
    dist = torch.norm(diff, dim=-1)                  # [B, B]

    # ----------- 3. Mask diagonal (self-distance) -----------
    B = X.size(0)
    dist[torch.arange(B), torch.arange(B)] = float("inf")

    # ----------- 4. kNN search -----------
    _, knn_idx = torch.topk(dist, k_neighbors, dim=1, largest=False)

    # ----------- 5. Gather neighbors -----------
    X_neighbors = X[knn_idx]    # [B, k, F]

    # ----------- 6. Remove UTM_X, UTM_Y from features -----------
    keep_indices = [i for i in range(len(features)) if i not in (ix, iy)]

    X_neighbors = X_neighbors[:, :, keep_indices]    # [B, k, F-2]

    return X_neighbors

class HybridGroupedSpatialMambaModel2(nn.Module):
    """
    Simplified multi-branch model with spatial input:
      - X_flat    : [B, F] - flat features split into groups
      - X_spatial : [B, K, F_spatial] - spatial neighbor features
    """
    def __init__(
        self,
        features,
        spatial_dim,
        d_model=64,
        group_d_model=32,
        spatial_d_model=64,
        dropout=0.3,  # Increased from 0.1
        n_layers=2    # Reduced from 4
    ):
        super().__init__()
        self.features = features
        self.dropout = nn.Dropout(dropout)
        self.name = 'HGSMamba2'
        self.spatial = True
        self.d_model = d_model
        
        # Define feature groups
        self.feature_groups = {
            "rainfall": ["Acc015_mm", "Acc030_mm", "Acc060_mm", "StormAccum_mm"],
            "intensity": ["Peak_I15_mm/h", "Peak_I30_mm/h", "Peak_I60_mm/h"],
            "geomorph": ["ContributingArea_km2", "PropHM23", "dNBR/1000", "KF"],
            "baseline": ["GaugeDist_m", "StormDur_H", "StormAvgI_mm/h"],
        }
        
        # Build index lists for each group
        self.group_indices = {}
        for group_name, group_feats in self.feature_groups.items():
            idxs = [features.index(f) for f in group_feats if f in features]
            if len(idxs) == 0:
                raise ValueError(f"Group {group_name} has no valid features.")
            self.group_indices[group_name] = idxs
        
        # Simple linear branch for each group with stronger regularization
        self.group_branches = nn.ModuleDict()
        for group_name, idxs in self.group_indices.items():
            in_dim = len(idxs)
            self.group_branches[group_name] = nn.Sequential(
                nn.Linear(in_dim, group_d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(group_d_model),  # Added normalization
            )
        
        self.spatial_input_proj = nn.Linear(spatial_dim, spatial_d_model)        
        self.spatial_layers = nn.ModuleList([
            Mamba(d_model=spatial_d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        self.spatial_norms = nn.ModuleList([
            nn.LayerNorm(spatial_d_model) 
            for _ in range(n_layers)
        ])
        
        total_dim = len(self.group_branches) * group_d_model + spatial_d_model
        
        self.combined_head = nn.Sequential(
            nn.Linear(total_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
    
    def split_features(self, x_flat):
        """Split flat features into groups"""
        group_features = {}
        for group_name, idxs in self.group_indices.items():
            group_features[group_name] = x_flat[:, idxs]
        return group_features
    
    def _spatial_forward(self, x_spatial):
        """
        Process spatial features with Mamba layers
        x_spatial: [B, K, F_spatial]
        """
        x = self.spatial_input_proj(x_spatial)  # [B, K, spatial_d_model]
        
        for layer, norm in zip(self.spatial_layers, self.spatial_norms):
            residual = x
            x = layer(x)
            x = norm(x)
            x = self.dropout(x)
            x = x + residual  # Residual connection
        
        x = x.mean(dim=1)  # Pool over neighbors: [B, spatial_d_model]
        return x
    
    def forward(self, x, target=None):
        """
        x: tuple of (x_flat, x_spatial)
          x_flat:    [B, F]
          x_spatial: [B, K, F_spatial]
        """
        x_flat, x_spatial = x
        
        # Process each feature group
        group_features = self.split_features(x_flat)
        group_outputs = []
        
        for group_name in self.group_indices.keys():
            group_x = group_features[group_name]
            out = self.group_branches[group_name](group_x)
            group_outputs.append(out)
        
        # Process spatial features
        spatial_out = self._spatial_forward(x_spatial)
        
        # Concatenate all outputs
        combined = torch.cat(group_outputs + [spatial_out], dim=1)  # [B, total_dim]
        
        # Final prediction
        output = self.combined_head(combined).squeeze(-1)
        probs = torch.sigmoid(output)
        
        if target is not None:
            loss = threat_score_loss(output, target)
        else:
            loss = None
        
        return probs, loss

from data import all_features

class SimpleMamba(nn.Module):
    """
    Simple Mamba model that uses only 4 features and node 0.
    Features: PropHM23, dNBR/1000, KF, Acc015_mm
    """
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2,):
        super().__init__()
        self.name = 'SimpleMamba'
        limited_features = ['PropHM23', 'dNBR/1000', 'KF', 'Acc015_mm', 'GaugeDist_m', 'ContributingArea_km2']
        self.feature_indices = [all_features.index(feat) for feat in limited_features if feat in all_features]

        self.n_features = len(self.feature_indices)  # PropHM23, dNBR/1000, KF, Acc015_mm
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Mamba block
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            #nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.loss_fn = nn.BCELoss()
    
    def forward(self, x, targets=None):
        """
        Args:
            x: Input tensor of shape (batch, 4) - only node 0 features
               Features in order: PropHM23, dNBR/1000, KF, Acc015_mm
            targets: Target tensor of shape (batch,)
        
        Returns:
            If targets provided: (loss, predictions)
            If targets None: predictions only
        """
        x = x[:, :, self.feature_indices]
        
        #x = x.unsqueeze(1)

        # Input projection
        x = self.input_proj(x)  # (batch, 1, d_model)
        x = self.norm1(x)
        
        # Mamba processing with residual
        #x = x + self.mamba(x)
        #x = self.norm2(x)
        
        # Feedforward with residual
        x = x + self.ff(x)
        x = self.norm3(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (batch, d_model)
        
        # Output prediction
        probs = self.output(x)  # (batch, n_classes)
        probs = probs[:, 0].squeeze(-1)  # (batch,)
        
        # Calculate loss if targets provided
        if targets is not None:
            loss = self.loss_fn(probs, targets)
            return probs, loss
        
        return probs, None
    
from data import all_features
from eval import ThreatScoreLoss

class ClusteredMambaModel_Flood(nn.Module):
    def __init__(self, pos_weight, input_dim=16, d_model=64, n_layers=4, dropout=0.1, n_clusters=1000):
        super().__init__()
        self.d_model = d_model
        self.name = 'ClusteredMamba_Flood'

        self.non_rainfall_features = [
            #'UTM_X', 'UTM_Y', 
            'GaugeDist_m', 
            'StormDur_H', 
            #'StormAccum_mm', 
            'StormAvgI_mm/h', 
            'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
            'ContributingArea_km2', 
            'PropHM23', 'dNBR/1000', 'KF', 'Missing_Data',
            #'Acc015_mm', 'Acc030_mm', 'Acc060_mm'
        ]

        self.rainfall_features = ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm']
        self.feature_names = self.non_rainfall_features + self.rainfall_features
        self.n_clusters = n_clusters
        
        # Get indices for rainfall vs non-rainfall features
        self.rainfall_indices = [all_features.index(feat) for feat in self.rainfall_features if feat in all_features]
        self.non_rainfall_indices = [all_features.index(feat) for feat in self.non_rainfall_features if feat in all_features]
        #self.non_rainfall_indices = [i for i in range(len(all_features)) if i not in self.rainfall_indices]

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
            #nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        #self.loss_fn = ThreatScoreLoss()
        
    def split_features(self, x):
        """Split input into rainfall and non-rainfall features"""
        # x shape: (batch_size, num_features)
        non_rainfall_x = x[:, 0, self.non_rainfall_indices]
        rainfall_x = x[:, 0, self.rainfall_indices]
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
    
    def forward(self, x, target=None):
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
        output = output.squeeze(-1)
        probs = torch.sigmoid(output)

        if target is not None:
            loss = self.loss_fn(output, target)
            return probs, loss

        return probs, None