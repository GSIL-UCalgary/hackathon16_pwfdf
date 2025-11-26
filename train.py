import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data import PWFDF_Data
from eval import evaluate, compare_params, evaluate_model, threat_score

from models.log_reg import Staley2017Model
from models.mamba import MambaClassifier, HybridMambaLogisticModel
from models.transformer import TransformerClassifier
from models.TSMixer import TSMixerClassifier

import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_file = './output/nandiff_logs.txt'

numerical_features = [
    #'UTM_X', 'UTM_Y', 
    'GaugeDist_m', 'StormDur_H', 'StormAccum_mm',
    'StormAvgI_mm/h', 'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
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
        logging.StreamHandler()  # This sends to console
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

def train_logistic(model, X_train, y_train, X_test, y_test, max_iter=1000):    
    print(f"Training on {len(y_train)} samples")
    print(f"Positive samples: {torch.sum(y_train).item()} ({100*torch.mean(y_train.float()):.1f}%)")
    print(f"Negative samples: {len(y_train) - torch.sum(y_train).item()} ({100*(1-torch.mean(y_train.float())):.1f}%)")
        
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
            
           
            print(f"Iter {iteration}: Loss={loss.item():.6f}, Test TS={test_ts:.4f}")
            '''
            print(f"  B={model.B.item():.4f}, Ct={model.Ct.item():.4f}, "
                  f"Cf={model.Cf.item():.4f}, Cs={model.Cs.item():.4f}")
            '''
        
        iteration += 1
        return loss
    
    # Run LBFGS optimization
    for epoch in range(max_iter):
        optimizer.step(closure)
        
        # Check for convergence
        if iteration >= max_iter:
            break
    
    print(f"Training completed after {iteration} iterations")
    
    return model

def train_mamba(model, X_train, y_train, X_test, y_test, max_epochs=100, patience=10):    
    model_save_path = f"./output/{model.name}_model.pth"

    print(f"Training on {len(y_train)} samples")
    print(f"Positive samples: {torch.sum(y_train).item()} ({100*torch.mean(y_train.float()):.1f}%)")
    print(f"Negative samples: {len(y_train) - torch.sum(y_train).item()} ({100*(1-torch.mean(y_train.float())):.1f}%)")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Count class distribution
    #n_neg = (y_train == 0).sum()
    #n_pos = (y_train == 1).sum()
    #pos_weight = n_neg / n_pos
    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    criterion = nn.BCELoss()
    
    train_losses = []
    test_metrics = []
    best_ts = 0.0
    best_epoch = 0
    patience_counter = 0
        
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        loss.backward()
        
        # Gradient clipping for stability
        #Torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
        scheduler.step(test_ts)
        
        # Print progress
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch:3d}: Loss={loss.item():.6f}, Train Acc={train_acc:.4f}")
            print(f"  Test TS={test_ts:.4f}, Test Acc={test_acc:.4f}, Test F1={test_f1:.4f}")
            print(f"  LR={optimizer.param_groups[0]['lr']:.2e}")
        
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
    
    # Load best model
    #model.load_state_dict(torch.load(model_save_path))
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
    
    training_results = {}
    test_results = {}
    
    # Approach 1: Classical Logistic
    print("=" * 60)
    print("APPROACH 1: Logistic Regression")
    print("=" * 60)
    setup_seed(42)
    model1 = Staley2017Model(numerical_features, duration='15min').to(device)
    model1 = train_logistic(model1, X_train, y_train, X_test, y_test, max_iter=100)
    training_results[model1.name] = evaluate_model(model1, X_train, y_train)
    test_results[model1.name] = evaluate_model(model1, X_test, y_test)

    # Approach 2: Mamba Feature Fusion
    print("\n" + "=" * 60)
    print("APPROACH 2: Mamba Feature Fusion")
    print("=" * 60)
    setup_seed(42)
    model2 = MambaClassifier(input_dim=X_train.shape[1], n_layers=2).to(device)
    model2 = train_mamba(model2, X_train, y_train, X_test, y_test, max_epochs=100)
    training_results[model2.name] = evaluate_model(model2, X_train, y_train)
    test_results[model2.name] = evaluate_model(model2, X_test, y_test)

    # Approach 3: Mamba × Rainfall Multiplication
    print("\n" + "=" * 60)
    print("APPROACH 3: Mamba × Rainfall Multiplication")
    print("=" * 60)
    setup_seed(42)
    model3 = HybridMambaLogisticModel(numerical_features, input_dim=X_train.shape[1], n_layers=2).to(device)
    model3 = train_mamba(model3, X_train, y_train, X_test, y_test, max_epochs=100)
    training_results[model3.name] = evaluate_model(model3, X_train, y_train)
    test_results[model3.name] = evaluate_model(model3, X_test, y_test)
    
    # Approach 4: TS Mixer
    print("\n" + "=" * 60)
    print("APPROACH 4: TS Mixer")
    print("=" * 60)
    setup_seed(42)
    model4 = TSMixerClassifier(input_dim=X_train.shape[1], d_model=64, n_layers=4, dropout=0.1).to(device)
    model4 = train_mamba(model4, X_train, y_train, X_test, y_test, max_epochs=100)
    training_results[model4.name] = evaluate_model(model4, X_train, y_train)
    test_results[model4.name] = evaluate_model(model4, X_test, y_test)

    print(test_results)

    # Compare results
    logging.info("\n" + "=" * 60)
    logging.info("COMPARISON SUMMARY")
    logging.info("=" * 60)
    logging.info("Train set")
    for approach, results in training_results.items():
        logging.info(f"{results['name']:25} TS: {results['ts']:.4f} | Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f}")
    logging.info("=" * 60)
    logging.info("Test set")
    for approach, results in test_results.items():
        logging.info(f"{results['name']:25} TS: {results['ts']:.4f} | Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | Recall: {results['recall']:.4f} | Precision: {results['precision']:.4f}")
    
    return test_results, [model1, model2, model3]

def main():
    data = PWFDF_Data()
    
    print(f"Total samples: {len(data.df)}")
    print(f"Training samples: {len(data.df[data.df['Database'] == 'Training'])}")
    print(f"Test samples: {len(data.df[data.df['Database'] == 'Test'])}\n")
    
    X_train, y_train, scaler = data.prepare_data_with_normalization(split='Training')
    X_test, y_test, _ = data.prepare_data_with_normalization(split='Test', scaler=scaler)

    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    X_test = torch.Tensor(X_test).to(device)
    y_test = torch.Tensor(y_test).to(device)

    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    #model = Staley2017Model().to(device)
    model = MambaClassifier(input_dim=X_train.shape[1]).to(device)
    #model = HybridMambaLogisticModel(input_dim=X_train.shape[1]).to(device)
    #model = TransformerClassifier(X_train.shape[1]).to(device)
    #model = TSMixerClassifier(input_dim=X_train.shape[1], d_model=64, n_layers=4, dropout=0.1)

    #model = train_lg(model, X_train, y_train, X_test, y_test, max_iter=100)
    model = train_mamba(model, X_train, y_train, X_test, y_test, max_epochs=100, patience=15)
    evaluate(model, X_test, y_test, logging)


if __name__ == "__main__":
    setup_seed(42)
    #main()
    results, models = compare_all_approaches()
