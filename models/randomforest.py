import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import torch
import torch.nn as nn

from eval import threat_score

from data import all_features

class RandomForestModel(nn.Module):
    """
    Random Forest wrapper that uses the same 4 features as Staley2017Model:
    - T (PropHM23): Proportion of high-moderate burn severity
    - F (dNBR/1000): Differenced Normalized Burn Ratio
    - S (KF): Kuchler Fire classification
    - R (Rainfall): Accumulated rainfall (duration-dependent)
    """
    
    def __init__(self, duration='15min', n_estimators=100, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                 random_state=None):
        super().__init__()
        self.name = 'RandomForest'

        limited_features = ['PropHM23', 'dNBR/1000', 'KF', 'Acc015_mm']
        #limited_features = ['Peak_I15_mm/h', 'StormAccum_mm', 'StormDur_H', 'PropHM23', 'ContributingArea_km2', 'KF', 'dNBR/1000']
        self.feature_indices = [all_features.index(feat) for feat in limited_features if feat in all_features]

        '''
        self.duration = duration
        self.spatial = False
        
        # Feature indices for T, F, S (same as Staley2017Model)
        self.T_idx = features.index('PropHM23')
        self.F_idx = features.index('dNBR/1000')
        self.S_idx = features.index('KF')
        
        # Rainfall index based on duration
        duration_map = {
            '15min': 'Acc015_mm',
            '30min': 'Acc030_mm',
            '60min': 'Acc060_mm'
        }

        feature_name = duration_map[duration]
        self.R_idx = features.index(feature_name) if feature_name in features else 0 # SETS TO 0 IF THAT FEATURE IS NOT PASSED IN
        '''

        # Store the 4 feature indices
        #self.feature_indices = [self.T_idx, self.F_idx, self.S_idx, self.R_idx]
        self.feature_names = limited_features
        
        # Initialize sklearn Random Forest
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.is_fitted = False
    
    def forward(self, x, target=None):
        """
        Predict probabilities for input x using only the 4 selected features.
        
        Args:
            x: torch.Tensor of shape (batch_size, num_features)
        
        Returns:
            torch.Tensor of predicted probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before forward pass")
        
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        
        x_selected = x_np[:, 0, self.feature_indices]
        probs = self.rf.predict_proba(x_selected)[:, 1]
        return torch.tensor(probs, dtype=torch.float32), None
    
    def fit(self, X, y):
        """
        Fit the Random Forest model using only the 4 selected features.
        
        Args:
            X: Features (torch.Tensor or numpy array)
            y: Labels (torch.Tensor or numpy array)
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
            
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = y
        
        X_selected = X_np[:, 0, self.feature_indices]
        self.rf.fit(X_selected, y_np)
        self.is_fitted = True
        
        return self
    
    def get_feature_importance(self):
        """Get feature importances from the trained model (for the 4 features)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        importances = self.rf.feature_importances_
        return dict(zip(self.feature_names, importances))

def train_random_forest(model: RandomForestModel, input_data):
    X_train, y_train, X_val, y_val = input_data
    model.fit(X_train, y_train)
    print(model.get_feature_importance())
    return model
