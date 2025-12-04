import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, seq_len, enc_in, d_model, dropout):
        super(ResBlock, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout)
        )
        self.channel = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, enc_in),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: [B, L, D]
        # Temporal mixing (across time/sequence dimension)
        #x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        # Channel mixing (across feature dimension)
        x = x + self.channel(x)
        return x


class TSMixerClassifier(nn.Module):
    def __init__(self, input_dim=14, d_model=64, n_layers=4, dropout=0.1):
        super(TSMixerClassifier, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.duration = '15min'
        self.name = 'TSMixer'
        
        # Stack of ResBlocks
        self.model = nn.ModuleList([
            ResBlock(seq_len=1, enc_in=input_dim, d_model=d_model, dropout=dropout) 
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, seq_len, input_dim] -> [B, seq_len * input_dim]
            nn.Linear(input_dim, 1),
            #nn.ReLU(),
            #nn.Dropout(dropout),
            #nn.Linear(d_model, 32),
            #nn.ReLU(),
            #nn.Dropout(dropout),
            #nn.Linear(d_model, 1),
            #nn.Sigmoid()
        )
        
    def forward(self, x, target=None):
        # x: [B, input_dim] for tabular data
        if len(x.shape) == 2:
            # Reshape to [B, seq_len, input_dim]
            x = x.unsqueeze(1)  # [B, 1, input_dim]
        
        for block in self.model:
            x = block(x)
        
        #if 

        return self.classifier(x).squeeze(-1)
    
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class SimpleHighPerformanceModel(nn.Module):
    """
    Simplified model focusing on what works:
    - Separate pathways for rainfall and environmental features
    - Minimal but effective architecture
    - Strong regularization
    """
    def __init__(self, features, input_dim=16, d_model=64, dropout=0.2):
        super().__init__()
        self.name = 'SimpleHybrid'
        self.duration = '15min'
        
        # Feature splits
        self.rainfall_features = ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm']
        self.all_features = features
        
        self.rainfall_indices = [features.index(f) for f in self.rainfall_features if f in features]
        self.non_rainfall_indices = [i for i in range(len(features)) if i not in self.rainfall_indices]
        
        n_rainfall = len(self.rainfall_indices)
        n_env = len(self.non_rainfall_indices)
        
        print(f"Rainfall features: {n_rainfall}, Environmental features: {n_env}")
        
        # === ENVIRONMENTAL PATHWAY (Mamba for complex patterns) ===
        self.env_encoder = nn.Sequential(
            nn.Linear(n_env, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Single Mamba layer (more is often worse for small datasets)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_norm = nn.LayerNorm(d_model)
        
        self.env_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # === RAINFALL PATHWAY (Simple but effective) ===
        self.rainfall_encoder = nn.Sequential(
            nn.Linear(n_rainfall, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.GELU()
        )
        
        # === FUSION ===
        fusion_dim = d_model // 2 + 16
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Split features
        env_x = x[:, self.non_rainfall_indices]
        rain_x = x[:, self.rainfall_indices]
        
        # Environmental pathway
        env = self.env_encoder(env_x)
        env = env.unsqueeze(1)  # Add sequence dimension
        env = self.mamba(env)
        env = self.mamba_norm(env)
        env = env.squeeze(1)
        env = self.env_head(env)
        
        # Rainfall pathway
        rain = self.rainfall_encoder(rain_x)
        
        # Combine and predict
        combined = torch.cat([env, rain], dim=1)
        output = self.fusion(combined)
        
        return output.squeeze(-1)


class UltraSimpleModel(nn.Module):
    """
    Even simpler: just a well-designed MLP with smart feature handling
    Often performs surprisingly well!
    """
    def __init__(self, features, input_dim=16, dropout=0.2):
        super().__init__()
        self.name = 'UltraSimple'
        self.duration = '15min'
        
        # Feature importance weighting
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Main network - deeper but narrow
        self.network = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Block 2
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Block 3
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Block 4
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Output
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply learned feature importance
        importance = self.feature_gate(x)
        x = x * importance
        
        # Main prediction
        return self.network(x).squeeze(-1)


class OptimalEnsemble(nn.Module):
    """
    Simple ensemble: Your best existing model + simple MLP
    Often gives best results with minimal complexity
    """
    def __init__(self, features, input_dim=16, d_model=64, dropout=0.2):
        super().__init__()
        self.name = 'OptimalEnsemble'
        self.duration = '15min'
        
        # Model 1: Hybrid approach
        self.model1 = SimpleHighPerformanceModel(features, input_dim, d_model, dropout)
        
        # Model 2: Simple MLP
        self.model2 = UltraSimpleModel(features, input_dim, dropout)
        
        # Learnable ensemble weight (start at 0.5/0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        
        # Weighted average with learned weight
        alpha = torch.sigmoid(self.alpha)
        return alpha * out1 + (1 - alpha) * out2


# === BONUS: Feature Engineering Helper ===
class FeatureEngineeringModel(nn.Module):
    """
    Simple model with explicit feature engineering
    Sometimes hand-crafted features beat deep learning!
    """
    def __init__(self, features, input_dim=16, dropout=0.2):
        super().__init__()
        self.name = 'FeatureEngineered'
        self.duration = '15min'
        
        self.all_features = features
        
        # Find key feature indices
        self.rainfall_idx = [i for i, f in enumerate(features) if 'Acc' in f or 'Storm' in f]
        self.intensity_idx = [i for i, f in enumerate(features) if 'Peak_I' in f]
        self.burn_idx = [i for i, f in enumerate(features) if 'dNBR' in f or 'PropHM' in f]
        
        # Engineered features dimension
        n_engineered = 5  # We'll create 5 interaction features
        
        self.network = nn.Sequential(
            nn.Linear(input_dim + n_engineered, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def engineer_features(self, x):
        """Create interaction features"""
        batch_size = x.size(0)
        engineered = []
        
        # Feature 1: Total rainfall intensity
        if self.rainfall_idx:
            rain_sum = x[:, self.rainfall_idx].sum(dim=1, keepdim=True)
            engineered.append(rain_sum)
        
        # Feature 2: Max peak intensity
        if self.intensity_idx:
            peak_max = x[:, self.intensity_idx].max(dim=1, keepdim=True)[0]
            engineered.append(peak_max)
        
        # Feature 3: Burn severity × rainfall (key interaction!)
        if self.burn_idx and self.rainfall_idx:
            burn_rain = (x[:, self.burn_idx].mean(dim=1, keepdim=True) * 
                        x[:, self.rainfall_idx].mean(dim=1, keepdim=True))
            engineered.append(burn_rain)
        
        # Feature 4: Intensity × rainfall duration ratio
        if self.intensity_idx and self.rainfall_idx:
            intensity_ratio = (x[:, self.intensity_idx].mean(dim=1, keepdim=True) / 
                              (x[:, self.rainfall_idx].mean(dim=1, keepdim=True) + 1e-6))
            engineered.append(intensity_ratio)
        
        # Feature 5: Early vs late rainfall ratio
        if len(self.rainfall_idx) >= 2:
            early_late = (x[:, self.rainfall_idx[0:1]] / 
                         (x[:, self.rainfall_idx[-1:]] + 1e-6))
            engineered.append(early_late)
        
        # Pad if needed
        while len(engineered) < 5:
            engineered.append(torch.zeros(batch_size, 1, device=x.device))
        
        return torch.cat(engineered, dim=1)
    
    def forward(self, x):
        # Add engineered features
        engineered = self.engineer_features(x)
        x_augmented = torch.cat([x, engineered], dim=1)
        
        return self.network(x_augmented).squeeze(-1)
    
class BestSimpleModel(nn.Module):
    """
    Optimized for generalization on small datasets
    Key: Strong regularization + simpler architecture
    """
    def __init__(self, features, input_dim=16, d_model=48, dropout=0.3):
        super().__init__()
        self.name = 'BestSimple'
        self.duration = '15min'
        
        # Feature splits
        self.rainfall_features = ['Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm']
        self.all_features = features
        
        self.rainfall_indices = [features.index(f) for f in self.rainfall_features if f in features]
        self.non_rainfall_indices = [i for i in range(len(features)) if i not in self.rainfall_indices]
        
        n_rainfall = len(self.rainfall_indices)
        n_env = len(self.non_rainfall_indices)
        
        print(f"Rainfall: {n_rainfall}, Environmental: {n_env}")
        
        # === ENVIRONMENTAL PATHWAY ===
        # Simpler to prevent overfitting
        self.env_net = nn.Sequential(
            nn.Linear(n_env, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # === RAINFALL PATHWAY ===
        # Keep it simple
        self.rain_net = nn.Sequential(
            nn.Linear(n_rainfall, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.GELU()
        )
        
        # === FUSION ===
        fusion_dim = d_model // 2 + 8
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        env_x = x[:, self.non_rainfall_indices]
        rain_x = x[:, self.rainfall_indices]
        
        env = self.env_net(env_x)
        rain = self.rain_net(rain_x)
        
        combined = torch.cat([env, rain], dim=1)
        return self.fusion(combined).squeeze(-1)


class TestBlock(nn.Module):
    def __init__(self, batch_size, dropout=0.1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(batch_size, batch_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(batch_size, 1),
        )

    def forward(self, x):
        return self.layer(x)

class Test(nn.Module):
    def __init__(self, features, batch_size, input_dim=16, d_model=48, dropout=0.3):
        super().__init__()
        self.name = 'TestModel'
        
        self.layers = [TestBlock(batch_size) for i in range(14)]


    def forward(self, x):
        for i in range(14):
            self.layers[i](x)

        # weighted summation of output of 14 layers

class TestBlock(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 1024), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, input_dim), 
        )

    def forward(self, x):
        # x is expected to have shape (actual_batch_size, input_dim)
        return self.layer(x)

class Test(nn.Module):
    def __init__(self, batch_size, num_blocks, dropout: float = 0.1):
        super().__init__()
        self.name = 'TestModel'

        self.layer1 = TestBlock(input_dim=batch_size, dropout=dropout) 
        self.layer2 = TestBlock(input_dim=batch_size, dropout=dropout) 
        #self.layer3 = nn.Linear(batch_size, 1)

        # Learnable weights for the weighted summation
        self.weights = nn.Parameter(torch.ones(16))


    def forward(self, x):
        # List to store the output of each TestBlock
        x = x.T
        block_outputs = []

        x = self.layer1(x)
        x = x + self.layer2(x)
        #x = self.layer3(x)
        
        print(x.shape) # N, B
        weights = self.weights.unsqueeze(0)
        print(weights.shape)

        weighted_sum = torch.matmul(weights, x)

        print(weighted_sum.shape)
        
        return weighted_sum.squeeze(0)

from models.mamba import threat_score_loss

class StaticMLPModel(nn.Module):
    def __init__(self, features, duration, d_model=64, dropout=0.1):
        super().__init__()
        self.name = 'StaticMLPModel'
        self.d_model = d_model
        
        # ... (Feature index finding remains the same for T, F, S, R) ...
        self.T_idx = features.index('PropHM23')
        self.F_idx = features.index('dNBR/1000')
        self.S_idx = features.index('KF')
        duration_map = {'15min': 'Acc015_mm', '30min': 'Acc030_mm', '60min': 'Acc060_mm'}
        feature_name = duration_map[duration]
        self.R_idx = features.index(feature_name) if feature_name in features else 0
        
        # 1. Feature Projection/Embedding (Replaces individual Mamba blocks)
        # We use simple Linear layers (Dense layers) for feature transformation
        self.proj_T = nn.Linear(1, d_model)
        self.proj_F = nn.Linear(1, d_model)
        self.proj_S = nn.Linear(1, d_model)
        self.proj_R = nn.Linear(1, d_model)
        
        # 2. Regularization
        self.dropout = nn.Dropout(dropout) 
        self.activation = nn.ReLU() # Added a non-linearity for the feature transformations
        
        # 3. Prediction Head (Replaces the Global Mamba + Pooling)
        # Input size is 4 * d_model (256) after concatenation
        self.prediction_head = nn.Sequential(
            nn.Linear(4 * d_model, d_model * 2), # Wider first layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1) # Final output
        )

    def forward(self, x, target=None):
        # x is (Batch, Features)
        
        # 1. Slicing and Transformation
        # x[:, self.T_idx] -> (B,)
        # x[:, self.T_idx].unsqueeze(-1) -> (B, 1) - Input shape for nn.Linear(1, d_model)

        x_T_in = x[:, self.T_idx].unsqueeze(-1)
        x_F_in = x[:, self.F_idx].unsqueeze(-1)
        x_S_in = x[:, self.S_idx].unsqueeze(-1)
        x_R_in = x[:, self.R_idx].unsqueeze(-1)

        # Apply projection, activation, and dropout
        x_T = self.dropout(self.activation(self.proj_T(x_T_in))) # Output: (B, d_model)
        x_F = self.dropout(self.activation(self.proj_F(x_F_in)))
        x_S = self.dropout(self.activation(self.proj_S(x_S_in)))
        x_R = self.dropout(self.activation(self.proj_R(x_R_in)))
        
        # 2. Concatenation and Fusion
        # Output: (B, 4 * d_model)
        x_fused = torch.cat([x_T, x_F, x_S, x_R], dim=-1) 
        
        # 3. Final Prediction
        output = self.prediction_head(x_fused)
        
        if target != None:
            #loss = self.criterion(output, target)
            loss = threat_score_loss(output, target)
        else:
            loss = None

        return torch.sigmoid(output), loss