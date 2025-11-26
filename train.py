import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from data import PWFDF_Data
from eval import evaluate, compare_params, evaluate_model, threat_score

from models.log_reg import Staley2017Model, LogisticRegression
from models.mamba import MambaClassifier, HybridMambaLogisticModel
from models.transformer import TransformerClassifier
from models.TSMixer import TSMixerClassifier, BestSimpleModel

import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_file = './output/seeds100_logs.txt'

numerical_features = [
    #'UTM_X', 'UTM_Y', 
    'GaugeDist_m', 
    'StormDur_H', 'StormAccum_mm', 'StormAvgI_mm/h', 
    'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
    'ContributingArea_km2', 
    'PropHM23', 'dNBR/1000', 'KF', 'Acc015_mm', 
    'Acc030_mm', 'Acc060_mm'
]

# Setup logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    #format='%(asctime)s - %(levelname)s - %(message)s',
    format='%(message)s',
    handlers=[
        logging.FileHandler(output_file, encoding='utf-8'),
        #logging.StreamHandler()  # This sends to console
    ]
)

# random seed setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_logistic(model, X_train, y_train, X_test, y_test, seed, max_iter=1000):    
    # Use LBFGS optimizer (same as sklearn's lbfgs)
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn='strong_wolfe'
    )
    
    criterion = nn.BCELoss()
    
    iteration = 0
    pbar = tqdm(total=max_iter, desc=f"Training {model.name} Seed={seed}", unit="iter", disable=False)

    def closure():
        nonlocal iteration
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        
        if iteration % 10 == 0:
            model.eval()
            with torch.no_grad():
                y_test_pred = model(X_test).cpu().numpy().flatten()
            test_ts = threat_score(y_test.cpu().numpy(), y_test_pred)
            model.train()
            
            #print(f"Iter {iteration}: Loss={loss.item():.6f}, Test TS={test_ts:.4f}")
            pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'Test TS': f'{test_ts:.4f}'})
            '''
            print(f"  B={model.B.item():.4f}, Ct={model.Ct.item():.4f}, "
                  f"Cf={model.Cf.item():.4f}, Cs={model.Cs.item():.4f}")
            '''
        
        iteration += 1
        pbar.update(1)
        return loss
    
    # Run LBFGS optimization
    for epoch in range(max_iter):
        optimizer.step(closure)
        
        # Check for convergence
        if iteration >= max_iter:
            break
    
    pbar.close()
    #print(f"Training completed after {iteration} iterations")
    
    return model

def train_logistic_fast(model, X_train, y_train, X_test, y_test, seed, max_iter=1000):    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    pbar = tqdm(total=max_iter, desc=f"Training {model.name} Seed={seed}", disable=True)
    
    for iteration in range(max_iter):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if iteration % 50 == 0:
            model.eval()
            with torch.no_grad():
                y_test_pred = model(X_test).cpu().numpy().flatten()
            test_ts = threat_score(y_test.cpu().numpy(), y_test_pred)
            pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'Test TS': f'{test_ts:.4f}'})
        
        pbar.update(1)
    
    pbar.close()
    return model

class ThreatScoreLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        # Use soft predictions for differentiability
        tp = torch.sum(y_pred * y_true)
        fp = torch.sum(y_pred * (1 - y_true))
        fn = torch.sum((1 - y_pred) * y_true)
        
        # Threat Score = TP / (TP + FP + FN)
        ts = tp / (tp + fp + fn + self.epsilon)
        
        # Return negative since we want to maximize TS
        return 1 - ts

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, y_pred, y_true):
        bce = nn.functional.binary_cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


def train_mamba(model, X_train, y_train, X_test, y_test, seed, max_epochs=200, patience=10):    
    model_save_path = f"./output/{model.name}_model.pth"
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    # Count class distribution
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = n_neg / n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    #criterion = nn.BCELoss()
    #criterion = FocalLoss(alpha=0.25, gamma=2.0)

    train_losses = []
    test_metrics = []
    best_ts = 0.0
    best_epoch = 0
    patience_counter = 0
        
    pbar = tqdm(range(max_epochs), desc=f"Training {model.name} Seed={seed}", unit="epoch", disable=True)
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        loss.backward()   
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Training metrics            
            train_pred = (y_pred > 0.5).float()
            train_acc = accuracy_score(y_train.cpu().numpy(), train_pred.cpu().numpy())
            
            # Test metrics
            y_test_pred = model(X_test).cpu().numpy().flatten()
            test_ts = threat_score(y_test.cpu().numpy(), y_test_pred)
            test_pred = (y_test_pred > 0.5).astype(int)
            test_acc = accuracy_score(y_test.cpu().numpy(), test_pred)
            test_f1 = f1_score(y_test.cpu().numpy(), test_pred)
            
        # Update learning rate
        #scheduler.step()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'TS': f'{test_ts:.4f}',
            'Acc': f'{test_acc:.4f}',
            'F1': f'{test_f1:.4f}',
            'Best': f'{best_ts:.4f}'
        })
        
        #if epoch % 10 == 0 or epoch == max_epochs - 1:
        #    tqdm.write(f"Epoch {epoch:3d}: Loss={loss.item():.6f}, Train Acc={train_acc:.4f}")
        #    tqdm.write(f"  Test TS={test_ts:.4f}, Test Acc={test_acc:.4f}, Test F1={test_f1:.4f}")
        #    tqdm.write(f"  LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if test_ts > best_ts:
            best_ts = test_ts
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            pass
            #print(f"Early stopping at epoch {epoch}")
            #break
            
        train_losses.append(loss.item())
        test_metrics.append({
            'epoch': epoch,
            'ts': test_ts,
            'acc': test_acc,
            'f1': test_f1
        })
    
    pbar.close()

    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    #print(f"Training completed. Best Test TS={best_ts:.4f} at epoch {best_epoch}")
    
    return model

def compare_all_approaches():
    data = PWFDF_Data()
    
    print(f"Total samples: {len(data.df)}")
    print(f"Training samples: {len(data.df[data.df['Database'] == 'Training'])}")
    print(f"Test samples: {len(data.df[data.df['Database'] == 'Test'])}\n")

    X_train, y_train, scaler = data.prepare_data_usgs(numerical_features, split='Training')
    X_test, y_test, _ = data.prepare_data_usgs(numerical_features, split='Test', scaler=scaler)

    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    X_test = torch.Tensor(X_test).to(device)
    y_test = torch.Tensor(y_test).to(device)

    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print(f"Training on {len(y_train)} samples")
    print(f"Positive samples: {torch.sum(y_train).item()} ({100*torch.mean(y_train.float()):.1f}%)")
    print(f"Negative samples: {len(y_train) - torch.sum(y_train).item()} ({100*(1-torch.mean(y_train.float())):.1f}%)")

    input_dim = X_train.shape[1]

    training_results = {}
    test_results = {}
        
    best_seed_for_model = {}   # { model_name: seed }
    best_metrics_for_model = {}  # { model_name: metrics dict }
    best_ts_for_model = {}     # { model_name: TS float }

    model_classes = [
        lambda: Staley2017Model(numerical_features, duration='15min'),
        lambda: LogisticRegression(numerical_features, duration='15min'),
        lambda: MambaClassifier(input_dim=input_dim, n_layers=2),
        lambda: HybridMambaLogisticModel(numerical_features, input_dim=input_dim, n_layers=1),
        lambda: TSMixerClassifier(input_dim=input_dim),
    ]

    for seed in tqdm(range(100), desc="Seeds", position=0):
    #for seed in range(100):
        #print(f"\n========== SEED {seed} ==========")

        for make_model in model_classes:
            setup_seed(seed)
            model = make_model().to(device)

            model = model.to(device)
            #print("=" * 60)
            #print(f"APPROACH 1: {model.name} ({seed})")
            #print("=" * 60)

            if model.name == 'Staley' or model.name == 'LogisticRegression':
                model = train_logistic_fast(model, X_train, y_train, X_test, y_test, seed, max_iter=100)
            else:
                model = train_mamba(model, X_train, y_train, X_test, y_test, seed, max_epochs=100)

            train_metrics = evaluate_model(model, X_train, y_train)
            test_metrics = evaluate_model(model, X_test, y_test)

            training_results[model.name] = train_metrics
            test_results[model.name] = test_metrics

            ts = test_metrics['ts']
            name = model.name

            # Initialize if needed
            if name not in best_ts_for_model:
                best_ts_for_model[name] = ts
                best_seed_for_model[name] = seed
                best_metrics_for_model[name] = test_metrics
            else:
                # Update if new TS is better
                if ts > best_ts_for_model[name]:
                    best_ts_for_model[name] = ts
                    best_seed_for_model[name] = seed
                    best_metrics_for_model[name] = test_metrics
            
            torch.cuda.empty_cache()

        #print(test_results)

        # Compare results
        logging.info("\n" + "=" * 60)
        logging.info(f"COMPARISON SUMMARY ({seed})")
        logging.info("=" * 60)
        logging.info("Train set")
        for approach, results in training_results.items():
            logging.info(f"{results['name']:25} TS: {results['ts']:.4f} | Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f}")
        logging.info("=" * 60)
        logging.info("Test set")
        for approach, results in test_results.items():
            logging.info(f"{results['name']:25} TS: {results['ts']:.4f} | Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | Recall: {results['recall']:.4f} | Precision: {results['precision']:.4f}")

    logging.info("\n========================= BEST SEEDS =========================")
    for name in best_seed_for_model:
        logging.info(f"\nModel: {name}")
        logging.info(f"  Best Seed: {best_seed_for_model[name]}")
        logging.info(f"  Best TS:   {best_ts_for_model[name]:.4f}")
        logging.info(f"  Metrics:   {best_metrics_for_model[name]}")


if __name__ == "__main__":
    setup_seed(42)
    compare_all_approaches()
