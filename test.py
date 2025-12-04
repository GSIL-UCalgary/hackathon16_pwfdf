import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as n

from sklearn.model_selection import train_test_split

import data
from data import PWFDF_Data
from eval import evaluate_model

torch.set_default_device('cuda') 

# random seed setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #torch.use_deterministic_algorithms(True)

setup_seed(42)
df_data = PWFDF_Data()

print(f"Total samples: {len(df_data.df)}")
print(f"Training samples: {len(df_data.df[df_data.df['Database'] == 'Training'])}")
print(f"Test samples: {len(df_data.df[df_data.df['Database'] == 'Test'])}\n")

X_train_full, y_train_full, scaler = df_data.prepare_data_usgs(data.all_features, split='Training')
#X_train, X_val, y_train, y_val = X_train_full, [], y_train_full, []
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full)
X_test, y_test, _ = df_data.prepare_data_usgs(data.all_features, split='Test', scaler=scaler)

X_train = torch.tensor(X_train)
X_val = torch.tensor(X_val)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train)
y_val = torch.tensor(y_val)
y_test = torch.tensor(y_test)

from models.mp_mamba import MultiPathwayHybridModel_og
from models.randomforest import RandomForestModel, train_random_forest

#model = MultiPathwayHybridModel_og(features=data.all_features, pos_weight=None, d_model=128, n_layers=6)
model = RandomForestModel(random_state=None)
model.load_state_dict(torch.load('/home/quinn/projects/pwfdf/output/best_models/RandomForest_best.pth')) # Load best model

results = evaluate_model(model, X_test, y_test)
print(f"{results['name']:25} TS: {results['ts']:.4f} | Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | Recall: {results['recall']:.4f} | Precision: {results['precision']:.4f}")
