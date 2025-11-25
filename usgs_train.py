import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from eval import find_best_threshold, evaluate, compare_params
from models.log_reg import Staley2017Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_data(pwfdf_data, duration='15min', split='Training'):
    """
    Prepare training data directly from PWFDF_Data using Database column
    
    Args:
        pwfdf_data: PWFDF_Data object
        duration: '15min', '30min', or '60min'
        split: 'Training' or 'Test'
    """
    df = pwfdf_data.df
    
    # Filter by Database column
    df = df[df['Database'] == split].copy()
    
    # Get all numerical features
    numerical_features = [
        'UTM_X', 'UTM_Y', 'GaugeDist_m', 'StormDur_H', 'StormAccum_mm',
        'StormAvgI_mm/h', 'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
        'ContributingArea_km2', 'PropHM23', 'dNBR/1000', 'KF',
        'Acc015_mm', 'Acc030_mm', 'Acc060_mm'
    ]
    
    X = df[numerical_features].values
    y = df['Response'].values

    # Extract features
    T = df['PropHM23'].values
    F = df['dNBR/1000'].values
    S = df['KF'].values
    
    # Get rainfall based on duration
    if duration == '15min':
        R = df['Acc015_mm'].values
    elif duration == '30min':
        R = df['Acc030_mm'].values
    else:  # 60min
        R = df['Acc060_mm'].values
    
    y = df['Response'].values
    
    # Remove rows with missing values
    mask = ~(np.isnan(T) | np.isnan(F) | np.isnan(S) | np.isnan(R) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    return torch.Tensor(X).to(device), torch.Tensor(y).to(device)

def train(model, X_train, y_train, X_test, y_test, max_iter=1000):    
    print(f"Training on {len(y_train)} samples")
    print(f"Positive samples: {torch.sum(y_train).item()} ({100*torch.mean(y_train.float()):.1f}%)")
    print(f"Negative samples: {len(y_train) - torch.sum(y_train).item()} ({100*(1-torch.mean(y_train.float())):.1f}%)")
    
    #y_train = y_train.reshape(-1, 1)
    
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
    
    # Training with LBFGS
    print(f"Training {model.duration} model with LBFGS...")
    
    iteration = 0
    
    def closure():
        nonlocal iteration
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        
        if iteration % 10 == 0:
            model.eval()
            with torch.no_grad():
                y_test_pred = model(X_test).cpu().numpy().flatten()
            threshold, test_ts = find_best_threshold(y_test.cpu().numpy(), y_test_pred)
            model.train()
            
            print(f"Iter {iteration}: Loss={loss.item():.6f}, Test TS={test_ts:.4f}")
            print(f"  B={model.B.item():.4f}, Ct={model.Ct.item():.4f}, "
                  f"Cf={model.Cf.item():.4f}, Cs={model.Cs.item():.4f}")
        
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

def main():
    from data import PWFDF_Data
    
    # Load data
    print("Loading PWFDF data...")
    data = PWFDF_Data()
    
    print(f"Total samples: {len(data.df)}")
    print(f"Training samples: {len(data.df[data.df['Database'] == 'Training'])}")
    print(f"Test samples: {len(data.df[data.df['Database'] == 'Test'])}\n")
    
    # Train models for each duration
    durations = ['15min', '30min', '60min']
    models = {}
    
    for duration in durations:
        print(f"\n{'='*60}")
        print(f"Training {duration} model")
        print(f"{'='*60}")
        
        X_train, y_train = prepare_data(data, duration=duration, split='Training')
        X_test, y_test = prepare_data(data, duration=duration, split='Test')

        model = Staley2017Model(duration=duration).to(device)
        model = train(model, X_train, y_train, X_test, y_test, max_iter=100)
        models[duration] = model
        
        print(f"\n{'='*50}")
        print(f"Test Set Results for {model.duration}")
        evaluate(model, X_test, y_test)
        print(f"\nLearned Parameters:")
        print(f"  B  = {model.B.item():.4f}")
        print(f"  Ct = {model.Ct.item():.4f}")
        print(f"  Cf = {model.Cf.item():.4f}")
        print(f"  Cs = {model.Cs.item():.4f}")
    
    # Compare with published parameters
    compare_params(models, durations)

if __name__ == "__main__":
    main()