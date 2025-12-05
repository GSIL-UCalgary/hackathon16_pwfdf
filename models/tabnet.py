import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

import globals


def cnp(x):
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = x

    return x_np

class TabNetModel(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.name = 'TabNetModel'
        self.clf = TabNetClassifier()
        self.feature_indices = [globals.all_features.index(feat) for feat in features if feat in globals.all_features]
        self.hyperparameters = {
            "None": 0
        }

    def forward(self, x, target=None):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before forward pass")
        
        x_np = cnp(x)
        x_selected = x_np[:, 0, self.feature_indices]
        probs = self.clf.predict(x_selected)
        #print(probs)
        return torch.tensor(probs, dtype=torch.float32), None

    def fit(self, X, y, X_val, y_val):
        X_np = cnp(X)
        y_np = cnp(y)
        X_val_np = cnp(X_val)
        y_val_np = cnp(y_val)

        X_selected = X_np[:, 0, self.feature_indices]
        X_val_selected = X_val_np[:, 0, self.feature_indices]

        self.clf.fit(X_selected, y_np, eval_set=[(X_val_selected, y_val_np)])
        self.is_fitted = True

def train_tabnet(model: TabNetModel, input_data):
    X_train, y_train, X_val, y_val, _, _ = input_data
    model.fit(X_train, y_train, X_val, y_val)
    return model
