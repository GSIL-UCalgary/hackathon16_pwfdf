import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product

from tqdm import tqdm
import wandb

import globals
from data import PWFDF_Data
from eval import evaluate_model, threat_score

import logging

#torch.set_default_device('cuda') 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_WANDB = True
WANDB_PROJECT='PWFDF'

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
    X_train, y_train, X_val, y_val, _, _ = input_data

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
    X_train, y_train, X_val, y_val, X_test, y_test = input_data
    model_save_path = f"./output/{model.name}_model.pth"

    config={
        "model": model.name,
        "seed": seed,
        "learning_rate": 1e-3,
        "optimizer": "AdamW",
        "features": used_features,
        "train size": len(X_train),
        "val size": len(X_val),
        "hyperparamters": model.hyperparameters,
    }

    if USE_WANDB:
        run = wandb.init( project=WANDB_PROJECT, name=f"{model.name}_run", config=config, reinit=True)
        wandb.watch(model, model.loss_fn, log="all", log_freq=100)

    # Hyperparameters for Early Stopping
    PATIENCE = 5 # The number of epochs to wait for improvement (Increased from 5 as recommended)
    MIN_DELTA = 1e-6 # Minimum improvement to count as "better" (e.g., 0.0001 TS improvement)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_metrics = evaluate_model(model, X_train, y_train)
            val_metrics = evaluate_model(model, X_val, y_val)
            test_metrics = evaluate_model(model, X_test, y_test)
            #test_metrics = evaluate_model(model, X_tes)
            #current_ts = -val_metrics['loss'].item()
            current_ts = val_metrics['ts']
            
        #scheduler.step(current_ts)
        #scheduler.step()

        # --- Early Stopping Logic ---
        if current_ts > best_ts + MIN_DELTA:
            # New best model found! Reset counter and save checkpoint.
            best_ts = current_ts
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path) # Save the state dictionary of the model
        elif current_ts != 0.0:
        #else:
            #print(current_ts)
            # Performance did not improve enough
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                pass
                #tqdm.write(f"\nEarly stopping at epoch {epoch}. Best TS was {best_ts:.4f}.")
                #break # Exit the training loop
        

        # --- Progress Bar Update ---
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'TS': f"{current_ts:.4f}",
            'Acc': f"{val_metrics['accuracy']:.4f}",
            'BestTS': f'{best_ts:.4f}',
            'Patience': f'{epochs_no_improve}/{PATIENCE}'
        })

        if USE_WANDB:
            wandb.log({
                "train_loss": train_metrics['loss'].item(),
                "train_ts": train_metrics['ts'],
                "train_acc": train_metrics['accuracy'],
                "val_loss": val_metrics["loss"].item(),
                "val_ts": val_metrics["ts"],
                "val_acc": val_metrics['accuracy'],
                "test_loss": test_metrics["loss"].item(),
                "test_ts": test_metrics["ts"],
                "test_acc": test_metrics['accuracy'],
                "best_ts": best_ts,
            })

    
    pbar.close()
    model.load_state_dict(torch.load(model_save_path)) # Load best model
    os.remove(model_save_path)

    return model

used_features2 = ['PropHM23', 'dNBR/1000', 'KF', 'Acc015_mm']
used_features4 = [
    #'Year', 
    'Latitude', 'Longitude', 
    'GaugeDist_m', 'ContributingArea_km2',
    'StormDur_H', 'StormMonth',
    
    'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h', 'StormAvgI_mm/h', 
    'PropHM23', 'dNBR/1000', 'KF', 'KF_Acc015',

    'Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm', 
]

used_features = [
    # Original features
    'Latitude', 'Longitude', 
    'GaugeDist_m', 'ContributingArea_km2',
    'StormDur_H', 'StormMonth',
    'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h', 'StormAvgI_mm/h', 
    'PropHM23', 'dNBR/1000', 'KF',
    'Acc015_mm', 'Acc030_mm', 'Acc060_mm', 'StormAccum_mm',
    
    # NEW: Engineered features
    'I15_to_Duration', 'I30_to_Duration', 'Peak_to_Avg_Ratio',
    'Burn_x_Peak', 'Burn_x_Accum', 'Burn_x_KF', 'HighMod_x_Peak',
    'KF_x_I15', 'KF_x_I30', 'KF_x_Accum',
    'Area_x_Distance', 'Area_per_Distance',
    'Early_to_Total_Ratio', 'Early_to_Late_Ratio',
    'Exceeds_I15_Threshold', 'Exceeds_I30_Threshold', 'High_Burn_Severity',
    'Storm_Energy', 'Erosion_Potential',
    'Is_Summer', 'Is_Monsoon'
]

from models.log_reg import Staley2017Model, LogisticRegression
from models.mamba import HybridMambaLogisticModel, MambaClassifier, ClusteredMambaModel_Flood
from models.mp_mamba import MultiPathwayHybridModel_og
#from models.transformer import TransformerClassifier, SimpleTransformerClassifier, AttentionClassifier
#from models.TSMixer import TSMixerClassifier, BestSimpleModel, Test, StaticMLPModel
from models.randomforest import RandomForestModel, train_random_forest
#from models.graph_mamba import GraphMambaModel, KNNMambaClassifier
#from models.fusion_mamba import MultiBranchMamba
#from models.graph import FixedNeighborhoodGNN
from models.simple import SimpleModel, SimpleAttention, SimpleMamba, FeatureGroupModel
from models.tabnet import TabNetModel, train_tabnet

def load_data():
    setup_seed(42)
    df_data = PWFDF_Data()
    
    print(f"Total samples: {len(df_data.df)}")
    print(f"Training samples: {len(df_data.df[df_data.df['Database'] == 'Training'])}")
    print(f"Test samples: {len(df_data.df[df_data.df['Database'] == 'Test'])}\n")

    X_train_full, y_train_full, scaler = df_data.prepare_data_usgs(globals.all_features, split='Training')
    X_train, X_val, y_train, y_val = X_train_full, [], y_train_full, []
    #X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)
    X_test, y_test, _ = df_data.prepare_data_usgs(globals.all_features, split='Test', scaler=scaler)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    X_train = torch.Tensor(X_train).to(device)
    X_val = torch.Tensor(X_val).to(device)
    X_test = torch.Tensor(X_test).to(device)

    y_train = torch.Tensor(y_train).to(device)
    y_val = torch.Tensor(y_val).to(device)
    y_test = torch.Tensor(y_test).to(device)

    for i, feature in enumerate(globals.all_features):
        print(f"{feature}: {X_train[0][0][i]}")

    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Val set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print(f"Training on {len(y_train)} samples")
    print(f"Positive samples: {torch.sum(y_train).item()} ({100*torch.mean(y_train.float()):.1f}%)")
    print(f"Negative samples: {len(y_train) - torch.sum(y_train).item()} ({100*(1-torch.mean(y_train.float())):.1f}%)")

    input_dim = X_train.shape[-1]
    print(f"Feature Size: {input_dim}")

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = n_neg / n_pos

    input_data = [X_train, y_train, X_val, y_val, X_test, y_test]

    return input_data, pos_weight

def compare_all_approaches():

    input_data, pos_weight = load_data()

    training_results = {}
    validation_results = {}
    test_results = {}
        
    best_seed_for_model = {}   # { model_name: seed }
    best_metrics_for_model = {}  # { model_name: metrics dict }
    best_ts_for_model = {}     # { model_name: TS float }
    worst_ts_for_model = {}     # { model_name: TS float }


    models = [
        lambda: Staley2017Model(duration='15min'),
        #lambda: SpatialMambaHybridModel(r_features, pos_weight),
        #lambda: SpatialMambaContextModel(r_features, pos_weight, 5 + 1)
        lambda: RandomForestModel(used_features, random_state=None),
        lambda: TabNetModel(used_features),
        #lambda: MambaClassifier(used_features, d_model=16, d_state=64, d_conv=4, expand=2, n_layers=6, dropout=0.2),
        #lambda: AttentionClassifier(used_features, d_model=128, n_heads=8, n_layers=3, dropout=0.3)
        lambda: SimpleModel(used_features, {"layers": 4, "d_model": 32, "dropout": 0.1, "hidden_dim": 8, "pos_weight": pos_weight,}),
        lambda: SimpleAttention(used_features, {"layers": 4, "d_model": 64, "dropout": 0.1, "hidden_dim": 8, "pos_weight": pos_weight,}),
        lambda: SimpleMamba(used_features, {"layers": 6, "d_model": 64, "dropout": 0.1, "hidden_dim": 8, "pos_weight": pos_weight,}),
        #lambda: FeatureGroupModel(pos_weight, d_model=32, dropout=0.1),
        #lambda: HybridMambaLogisticModel(used_features, pos_weight, d_model=128,  d_state=16, d_conv=4, expand=2, n_layers=4, dropout=0.1),
        #lambda: MultiPathwayHybridModel_og(used_features, d_model=128, n_layers=6, dropout=0.1),
        #lambda: ClusteredMambaModel_Flood(pos_weight, input_dim=input_dim, n_layers=1),
        #lambda: SimpleTransformerClassifier(),
        #lambda: HybridMambaFeatureGroups(features, pos_weight),
        #lambda: HybridGroupedSpatialMambaModel(features=features, spatial_dim=16),
        #lambda: HybridGroupedSpatialMambaModel2(features=features, spatial_dim=16),
        #lambda: GraphMambaModel(max_neighbors=5, hidden_dim=16),
        #lambda: KNNMambaClassifier(all_features=data.all_features, pos_weight=pos_weight, d_model=64, n_layers=2), # <-- ADD THIS LINE
        #lambda: MultiBranchMamba(feature_names=data.all_features, d_model=16, d_state=16, n_classes=1, fusion_method='attention'),
        #lambda: SimpleMamba(d_model=128, d_state=64, d_conv=4, expand=2),
        #lambda: FixedNeighborhoodGNN(len(data.all_features), hidden_features=32),
        #lambda: SimplifiedMPHModel()
        #lambda: HybridMambaMLPModel(),
    ]

    all_test_results = {model().name: [] for model in models}

    #seeds = [1, 42]
    #seeds = [0, 10, 42, 51]
    #seeds = range(0, 1)
    seeds = [42]
    epochs = 500
    #input_data = [X_train, y_train, X_test, y_test]

    X_train, y_train, X_val, y_val, X_test, y_test = input_data

    logging.info("\n\n\n========================= Comparison =========================")
    #logging.info(f"Total samples: {len(y_train + y_val + y_test)}")
    logging.info(f"Training set: {X_train.shape[0]} samples")
    logging.info(f"Val set: {X_val.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")
    logging.info(f"Input_data: {input_data[0].shape[0]}, {input_data[1].shape[0]}, {input_data[2].shape[0]}, {input_data[3].shape[0]}")
    logging.info(f"Seeds: {seeds}")
    logging.info(f"Features Count: {len(globals.all_features)}")
    logging.info(f"{globals.all_features}")

    for seed in tqdm(seeds, desc="Seeds", position=0):
        for make_model in models:
            setup_seed(seed)
            model = make_model().to(device)

            if model.name == 'Staley' or model.name == 'LogisticRegression':
                model = train_logistic(model, input_data, seed, max_iter=epochs)
            elif model.name == 'RandomForest':
                model = train_random_forest(model, input_data)
            elif model.name == 'TabNetModel':
                model = train_tabnet(model, input_data)
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

        print_full = True
        if print_full:
            logging.info("\n" + "=" * 60)
            logging.info(f"SUMMARY (Seed = {seed})")
            logging.info("=" * 60)
            logging.info("Train set")
            for approach, results in training_results.items():
                logging.info(f"{results['name']:25} TS: {results['ts']:.4f} | Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | Recall: {results['recall']:.4f} | Precision: {results['precision']:.4f}")
            logging.info("=" * 60)
            logging.info("Val set")
            for approach, results in validation_results.items():
                logging.info(f"{results['name']:25} TS: {results['ts']:.4f} | Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | Recall: {results['recall']:.4f} | Precision: {results['precision']:.4f}")
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


    #
    # Comparing Seeds
    #

    if False:
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

def get_model_factory(model_name):
    """Maps model name to its constructor."""
    if model_name == "SimpleMamba": return SimpleMamba
    if model_name == "SimpleModel": return SimpleModel
    if model_name == "SimpleAttention": return SimpleAttention
    if model_name == "RandomForest": return RandomForestModel
    if model_name == "Staley2017Model": return lambda: Staley2017Model(duration='15min')
    return None

def run_hyperparameter_sweep(model_name, input_data, pos_weight, sweep_config, seeds, epochs):
    """
    Performs a grid search for a single model and returns the best configuration.
    Logs each run to WandB.
    """
    logging.info(f"\n--- Starting Sweep for {model_name} ---")
    best_ts = -1.0
    best_hyperparameters = None
    X_train, y_train, X_val, y_val, X_test, y_test = input_data

    # Generate all combinations of hyperparameters
    keys, values = zip(*sweep_config.items())
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    make_model = get_model_factory(model_name)
    if not make_model:
        logging.error(f"Model factory not found for {model_name}")
        return None

    # Run sweep over all combinations
    for combo in tqdm(all_combinations, desc=f"Sweeping {model_name}"):
        
        wandb.init(project=WANDB_PROJECT, group=f"{model_name}_sweep", config=combo, reinit=True)
        hyperparameters = {**combo, "pos_weight": pos_weight}
        
        # Average metrics over all seeds for robust comparison (using only the first seed for this example)
        seed = seeds[0] 
        setup_seed(seed)
        
        # 1. Create Model
        model = make_model(used_features, hyperparameters)
        model = model.to(device)

        # 2. Train Model
        if model.name == 'Staley' or model.name == 'LogisticRegression':
            model = train_logistic(model, input_data, seed, max_iter=epochs)
        elif model.name == 'RandomForest':
            model = train_random_forest(model, input_data)
        else:
            model = train_mamba(seed, model, input_data, max_epochs=epochs)

        # 3. Evaluate on Validation Set (best practice for sweep)
        val_metrics = evaluate_model(model, X_val, y_val)
        
        # 4. Log results to WandB
        wandb.log({"val/ts": val_metrics['ts'], "val/accuracy": val_metrics['accuracy'], **hyperparameters})
        
        # 5. Check for best result based on Threat Score (ts)
        if val_metrics['ts'] > best_ts:
            best_ts = val_metrics['ts']
            best_hyperparameters = hyperparameters
            logging.info(f"New Best TS: {best_ts:.4f} with HPs: {combo}")

        wandb.finish()

    return best_hyperparameters

def final_train_with_best_hps(model_name, input_data, best_hps, seeds, epochs):
    """
    Performs the final training run using the optimal hyperparameters found.
    Logs the final test metrics to a separate WandB run.
    """
    logging.info(f"\n--- Final Training for {model_name} with Best HPs ---")
    X_train, y_train, X_val, y_val, X_test, y_test = input_data

    # Initialize WandB run for the final model
    wandb.init(project="my-tabular-project", name=f"FINAL_{model_name}", config=best_hps, reinit=True)

    make_model = get_model_factory(model_name)
    if not make_model: return

    final_test_results = []
    
    for seed in seeds:
        setup_seed(seed)
        
        # 1. Create Model with Best HPs
        # Note: We pass the full HPs including pos_weight
        model = make_model(used_features, best_hps)
        model = model.to(device)

        # 2. Train Model
        if model.name == 'Staley' or model.name == 'LogisticRegression':
            model = train_logistic(model, input_data, seed, max_iter=epochs)
        elif model.name == 'RandomForest':
            model = train_random_forest(model, input_data)
        else:
            model = train_mamba(seed, model, input_data, max_epochs=epochs)
        
        # 3. Evaluate on Test Set
        test_metrics = evaluate_model(model, X_test, y_test)
        final_test_results.append(test_metrics)
        
        # Log test metrics per seed
        wandb.log({
            f"test_seed_{seed}/ts": test_metrics['ts'],
            f"test_seed_{seed}/accuracy": test_metrics['accuracy'],
            f"test_seed_{seed}/f1": test_metrics['f1'],
        })

    # Calculate and log average test metrics across seeds
    avg_ts = sum(m['ts'] for m in final_test_results) / len(final_test_results)
    
    wandb.run.summary["avg_test_ts"] = avg_ts
    logging.info(f"Final Model: {model_name} | Average Test TS: {avg_ts:.4f}")
    
    wandb.finish()
    return avg_ts

def do_hyperparameter():
    input_data, pos_weight = load_data()

    SWEEP_CONFIGS = {
        "SimpleMamba": {
            "layers": [2, 4, 6],
            "d_model": [16, 32, 64],
            "dropout": [0.1, 0.2],
        },
        "SimpleModel": {
            "layers": [4, 8],
            "d_model": [16, 32],
            "dropout": [0.1, 0.3],
        },
        "SimpleAttention": {
            "layers": [2, 4],
            "d_model": [32, 64],
            "dropout": [0.1, 0.2],
        }
        # Other models (Staley2017Model, RandomForestModel) do not have hyperparameters defined here
    }

    seeds = [0]
    epochs = 500
    all_best_hps = {}

    # 1. Hyperparameter Sweep Stage
    for model_name, sweep_config in SWEEP_CONFIGS.items():
        best_hps = run_hyperparameter_sweep(model_name, input_data, pos_weight, sweep_config, seeds=[seeds[0]], epochs=epochs)
        
        if best_hps:
            all_best_hps[model_name] = best_hps
            logging.info(f"Found BEST HPs for {model_name}: {best_hps}")
        else:
            logging.warning(f"Could not find best HPs for {model_name}")

    # 2. Final Training and Logging Stage
    final_leaderboard = {}
    for model_name, best_hps in all_best_hps.items():
        # Train with best HPs across all specified seeds and log to WandB
        avg_ts = final_train_with_best_hps(model_name, input_data, best_hps, seeds, epochs)
        final_leaderboard[model_name] = avg_ts

    logging.info("\n\n========================= FINAL LEADERBOARD =========================")
    for model_name, avg_ts in sorted(final_leaderboard.items(), key=lambda item: item[1], reverse=True):
        logging.info(f"{model_name:25} | Avg Test TS: {avg_ts:.4f}")

if __name__ == "__main__":

    output_file = './output/logs/hps_sweep2.txt'

    # Setup logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(output_file, encoding='utf-8'),
            #logging.StreamHandler()  # This sends to console
        ]
    )

    compare_all_approaches()
    #do_hyperparameter()
