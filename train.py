import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import data
from data import PWFDF_Data
from eval import evaluate_model, threat_score

import logging

#torch.set_default_device('cuda') 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def train_logistic(model, input_data, seed, max_iter=1000):    
    X_train, y_train, X_val, y_val = input_data

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
    pbar = tqdm(total=max_iter, desc=f"Training {model.name} Seed={seed}", unit="iter", disable=True)

    def closure():
        nonlocal iteration
        model.train()
        optimizer.zero_grad()
        y_pred, _ = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        
        if iteration % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_metrics = evaluate_model(model, X_val, y_val)
            val_ts = val_metrics['ts']
            model.train()
            
            pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'Val TS': f'{val_ts:.4f}'})
        
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
    return model


def train_mamba(seed, model, input_data, max_epochs=200):
    X_train, y_train, X_val, y_val = input_data
    model_save_path = f"./output/{model.name}_model.pth"

    # Hyperparameters for Early Stopping
    PATIENCE = 15 # The number of epochs to wait for improvement (Increased from 5 as recommended)
    MIN_DELTA = 1e-4 # Minimum improvement to count as "better" (e.g., 0.0001 TS improvement)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

    train_losses = []
    val_metrics = []
    
    best_ts = -float('inf') # Initialize to negative infinity to ensure any TS is an improvement
    epochs_no_improve = 0
        
    pbar = tqdm(range(max_epochs), desc=f"Training {model.name} Seed={seed}", unit="epoch", disable=False)

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        y_pred, loss = model(X_train, y_train)
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            val_metrics = evaluate_model(model, X_val, y_val)
            current_ts = val_metrics['ts']
            
        scheduler.step(current_ts) # Update learning rate

        # --- Early Stopping Logic ---
        if current_ts > best_ts + MIN_DELTA:
            # New best model found! Reset counter and save checkpoint.
            best_ts = current_ts
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path) # Save the state dictionary of the model
        else:
            # Performance did not improve enough
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                tqdm.write(f"\nEarly stopping at epoch {epoch}. Best TS was {best_ts:.4f}.")
                break # Exit the training loop
                
        # --- Progress Bar Update ---
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'TS': f"{current_ts:.4f}",
            'Acc': f"{val_metrics['accuracy']:.4f}",
            'BestTS': f'{best_ts:.4f}',
            'Patience': f'{epochs_no_improve}/{PATIENCE}'
        })

    
    pbar.close()
    model.load_state_dict(torch.load(model_save_path)) # Load best model
    os.remove(model_save_path)

    return model

def compare_all_approaches():
    setup_seed(42)
    df_data = PWFDF_Data()
    
    print(f"Total samples: {len(df_data.df)}")
    print(f"Training samples: {len(df_data.df[df_data.df['Database'] == 'Training'])}")
    print(f"Test samples: {len(df_data.df[df_data.df['Database'] == 'Test'])}\n")

    X_train_full, y_train_full, scaler = df_data.prepare_data_usgs(data.all_features, split='Training')
    #X_train, X_val, y_train, y_val = X_train_full, [], y_train_full, []
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full)
    X_test, y_test, _ = df_data.prepare_data_usgs(data.all_features, split='Test', scaler=scaler)
    #X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.1, random_state=42, stratify=y_test)

    X_train = torch.Tensor(X_train).to(device)
    X_val = torch.Tensor(X_val).to(device)
    X_test = torch.Tensor(X_test).to(device)

    y_train = torch.Tensor(y_train).to(device)
    y_val = torch.Tensor(y_val).to(device)
    y_test = torch.Tensor(y_test).to(device)

    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Val set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print(f"Training on {len(y_train)} samples")
    print(f"Positive samples: {torch.sum(y_train).item()} ({100*torch.mean(y_train.float()):.1f}%)")
    print(f"Negative samples: {len(y_train) - torch.sum(y_train).item()} ({100*(1-torch.mean(y_train.float())):.1f}%)")

    input_dim = X_train.shape[-1]
    print(f"Feature Size: {input_dim}")

    training_results = {}
    validation_results = {}
    test_results = {}
        
    best_seed_for_model = {}   # { model_name: seed }
    best_metrics_for_model = {}  # { model_name: metrics dict }
    best_ts_for_model = {}     # { model_name: TS float }
    worst_ts_for_model = {}     # { model_name: TS float }

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = n_neg / n_pos

    output_file = './output/logs/new_seeds100.txt'

    # Setup logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(output_file, encoding='utf-8'),
            #logging.StreamHandler()  # This sends to console
        ]
    )

    from models.log_reg import Staley2017Model, LogisticRegression
    from models.mamba import SimpleMamba, ClusteredMambaModel_Flood
    from models.new_mamba import MultiBranchWithGlobalMamba
    from models.mp_mamba import MultiPathwayHybridModel_og
    from models.transformer import TransformerClassifier
    from models.TSMixer import TSMixerClassifier, BestSimpleModel, Test, StaticMLPModel
    from models.randomforest import RandomForestModel, train_random_forest
    from models.graph_mamba import GraphMambaModel, KNNMambaClassifier
    from models.fusion_mamba import MultiBranchMamba
    from models.graph import FixedNeighborhoodGNN

    models = [
        #lambda: Staley2017Model(data.all_features, duration='15min'),
        #lambda: SpatialMambaHybridModel(r_features, pos_weight),
        #lambda: SpatialMambaContextModel(r_features, pos_weight, 5 + 1)
        lambda: RandomForestModel(random_state=None),
        #lambda: HybridMambaLogisticModel(features, pos_weight, input_dim=input_dim, n_layers=1),
        #lambda: MultiPathwayHybridModel_og(features=data.all_features, d_model=128, n_layers=1),
        lambda: ClusteredMambaModel_Flood(pos_weight, input_dim=input_dim, n_layers=1),
        #lambda: HybridMambaFeatureGroups(features, pos_weight),
        #lambda: HybridGroupedSpatialMambaModel(features=features, spatial_dim=16),
        #lambda: HybridGroupedSpatialMambaModel2(features=features, spatial_dim=16),
        #lambda: GraphMambaModel(max_neighbors=5, hidden_dim=128),
        #lambda: KNNMambaClassifier(all_features=data.all_features, pos_weight=pos_weight, d_model=64, n_layers=2), # <-- ADD THIS LINE
        #lambda: MultiBranchMamba(feature_names=data.all_features, d_model=16, d_state=16, n_classes=1, fusion_method='attention'),
        #lambda: SimpleMamba(d_model=256, d_state=32),
        #lambda: FixedNeighborhoodGNN(len(data.all_features), hidden_features=32),
        #lambda: SimplifiedMPHModel()
    ]

    all_test_results = {model().name: [] for model in models}

    #seeds = [1, 42]
    #seeds = [0, 10, 42, 51]
    seeds = range(0, 100)
    epochs = 100
    input_data = [X_train, y_train, X_val, y_val]

    logging.info("\n\n\n========================= Comparison =========================")
    logging.info(f"Total samples: {len(df_data.df)}")
    logging.info(f"Training set: {X_train.shape[0]} samples")
    logging.info(f"Val set: {X_val.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")
    logging.info(f"Input_data: {input_data[0].shape[0]}, {input_data[1].shape[0]}, {input_data[2].shape[0]}, {input_data[3].shape[0]}")
    logging.info(f"Seeds: {seeds}")
    logging.info(f"Features Count: {len(data.all_features)}")
    logging.info(f"{data.all_features}")

    for seed in tqdm(seeds, desc="Seeds", position=0):
        for make_model in models:
            setup_seed(seed)
            model = make_model().to(device)

            if model.name == 'Staley' or model.name == 'LogisticRegression':
                model = train_logistic(model, input_data, seed, max_iter=epochs)
            elif model.name == 'RandomForest':
                model = train_random_forest(model, input_data)
            else:
                model = train_mamba(seed, model, input_data, max_epochs=epochs)

            training_results[model.name] = evaluate_model(model, X_train, y_train)

            if len(X_val) > 0:
                validation_results[model.name] = evaluate_model(model, X_val, y_val)

            test_metrics = evaluate_model(model, X_test, y_test)
            all_test_results[model.name].append(test_metrics)
            test_results[model.name] = test_metrics

            ts = test_metrics['ts']
            name = model.name

            if name not in best_ts_for_model or ts > best_ts_for_model[name]:
                best_ts_for_model[name] = ts
                best_seed_for_model[name] = seed
                best_metrics_for_model[name] = test_metrics
                
                os.makedirs('./output/best_models', exist_ok=True)
                torch.save(model.state_dict(), f'./output/best_models/{name}_best.pth')
            
            if name not in worst_ts_for_model or ts < worst_ts_for_model[name]:
                worst_ts_for_model[name] = ts

            torch.cuda.empty_cache()

        print_full = False
        if print_full:
            logging.info("\n" + "=" * 60)
            logging.info(f"SUMMARY (Seed = {seed})")
            logging.info("=" * 60)
            logging.info("Train set")
            for approach, results in training_results.items():
                logging.info(f"{results['name']:25} TS: {results['ts']:.4f} | Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f}")
            logging.info("=" * 60)
            logging.info("Test set")
            for approach, results in test_results.items():
                logging.info(f"{results['name']:25} TS: {results['ts']:.4f} | Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | Recall: {results['recall']:.4f} | Precision: {results['precision']:.4f}")
        else:
            logging.info(
                f"Seed={seed} | "
                + "Train: " + " | ".join([f"{r['name']}: {r['ts']:.4f}" for r in training_results.values()])
                + " || Val: " + " | ".join([f"{r['name']}: {r['ts']:.4f}" for r in validation_results.values()])
                + " || Test: " + " | ".join([f"{r['name']}: {r['ts']:.4f}" for r in test_results.values()])
            )
    logging.info("\n\n==================== AGGREGATE RESULTS =====================")
    logging.info(f"Summary over {len(seeds)} seeds on the Test Set:")
    
    for name, results_list in all_test_results.items():
        if not results_list:
            continue
            
        # Get all metric values for this model
        ts_scores = np.array([res['ts'] for res in results_list])
        acc_scores = np.array([res['accuracy'] for res in results_list])
        f1_scores = np.array([res['f1'] for res in results_list])
        recall_scores = np.array([res['recall'] for res in results_list])
        precision_scores = np.array([res['precision'] for res in results_list])

        # Calculate Mean and Standard Deviation
        ts_mean, ts_std = np.mean(ts_scores), np.std(ts_scores)
        acc_mean, acc_std = np.mean(acc_scores), np.std(acc_scores)
        f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
        recall_mean, recall_std = np.mean(recall_scores), np.std(recall_scores)
        precision_mean, precision_std = np.mean(precision_scores), np.std(precision_scores)
        
        # Report the results
        logging.info("\n" + "-" * 60)
        logging.info(f"Model: {name}")
        logging.info(f"  Threat Score (TS):  {ts_mean:.4f} ± {ts_std:.4f} (Best: {best_ts_for_model[name]:.4f}, Worst: {worst_ts_for_model[name]:.4f})")
        logging.info(f"  Accuracy (Acc):     {acc_mean:.4f} ± {acc_std:.4f}")
        logging.info(f"  F1 Score (F1):      {f1_mean:.4f} ± {f1_std:.4f}")
        logging.info(f"  Recall:             {recall_mean:.4f} ± {recall_std:.4f}")
        logging.info(f"  Precision:          {precision_mean:.4f} ± {precision_std:.4f}")

    logging.info("\n========================= BEST SEEDS =========================")
    for name in best_seed_for_model:
        logging.info(f"\nModel: {name}")
        logging.info(f"  Best Seed:  {best_seed_for_model[name]}")
        logging.info(f"  Best TS:    {best_ts_for_model[name]:.4f}")
        logging.info(f"  Best Worst: {worst_ts_for_model[name]:.4f}")
        logging.info(f"  Metrics:    {best_metrics_for_model[name]}")


if __name__ == "__main__":
    compare_all_approaches()
