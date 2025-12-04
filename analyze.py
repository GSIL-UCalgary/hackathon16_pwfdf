import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data import PWFDF_Data
import data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_folder = './output/'

def explain_mamba_model(model, X_train, X_test, feature_names, background_size=100):
    """
    Use SHAP to explain Mamba model predictions
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Create a wrapper function for SHAP
    def model_predict(x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(device)
        
        with torch.no_grad():
            pred, _ = model(x)
            return pred.cpu().numpy()
    
    # Select background data (subset of training data)
    if background_size > len(X_train):
        background_size = len(X_train)
    
    background_data = X_train[np.random.choice(len(X_train), background_size, replace=False)]
    
    print(f"Creating SHAP explainer with {background_size} background samples...")
    
    # Create SHAP explainer
    explainer = shap.Explainer(model_predict, background_data, feature_names=feature_names)
    
    # Calculate SHAP values for test set
    print("Calculating SHAP values...")
    shap_values = explainer(X_test)
    
    return shap_values, explainer

def plot_shap_summary(shap_values, feature_names, max_display=15):
    """Plot SHAP summary plot"""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_folder + 'shap_summary.png')

def plot_shap_bar(shap_values, feature_names, max_display=15):
    """Plot SHAP bar plot (mean absolute SHAP values)"""
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_folder + 'shap_bar.png')

def plot_feature_importance(shap_values, feature_names, top_k=10):
    """Custom feature importance plot"""
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=True)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'].tail(top_k), importance_df['importance'].tail(top_k))
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Top {top_k} Feature Importance (SHAP)')
    plt.tight_layout()
    plt.savefig(output_folder + 'shap_feature_importance.png')
    
    return importance_df

def analyze_individual_predictions(shap_values, X_test, feature_names, 
                                 sample_indices=None, n_samples=3):
    """Analyze individual predictions"""
    if sample_indices is None:
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        print(f"\n{'='*50}")
        print(f"Sample {i+1} (Index {idx})")
        print(f"{'='*50}")
        
        # Force plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
        plt.title(f"SHAP Explanation for Sample {idx}")
        plt.tight_layout()
        plt.savefig(output_folder + 'shap_individual_predictions.png')
        
        # Print feature values
        print("Feature values for this sample:")
        for j, feature in enumerate(feature_names):
            print(f"  {feature}: {X_test[idx, j]:.4f}")

def analyze_feature_dependencies(shap_values, X_test, feature_names, target_features=None):
    """Analyze dependencies between features"""
    if target_features is None:
        # Use top 5 most important features
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-5:]
        target_features = [feature_names[i] for i in top_indices]
    
    for feature in target_features:
        if feature in feature_names:
            feature_idx = feature_names.index(feature)
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_idx, 
                shap_values.values, 
                X_test, 
                feature_names=feature_names,
                show=False
            )
            plt.title(f"SHAP Dependence Plot for {feature}")
            plt.tight_layout()
            plt.savefig(output_folder + 'shap_feature_dependencies.png')

def compare_pathway_importance(model, X_test, feature_names):
    """
    Compare importance of Mamba pathway vs Logistic pathway
    """
    model.eval()
    
    # Get pathway contributions
    mamba_contributions = []
    logistic_contributions = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            x = torch.FloatTensor(X_test[i:i+1]).to(device)
            
            if hasattr(model, 'split_features'):
                non_rainfall_x, rainfall_x = model.split_features(x)
                
                mamba_out = model._mamba_forward(non_rainfall_x)
                logistic_out = model.logistic_layer(rainfall_x)
                combined = torch.cat([mamba_out, logistic_out], dim=1)
                final_pred = model.combined_head(combined)
                
                mamba_contributions.append(mamba_out.mean().item())
                logistic_contributions.append(logistic_out.mean().item())
    
    # Plot pathway contributions
    plt.figure(figsize=(10, 6))
    samples = range(len(mamba_contributions))
    plt.plot(samples, mamba_contributions, label='Mamba Pathway', alpha=0.7)
    plt.plot(samples, logistic_contributions, label='Logistic Pathway', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Pathway Contribution')
    plt.title('Mamba vs Logistic Pathway Contributions')
    plt.legend()
    plt.savefig(output_folder + 'shap_pathway_importance.png')
    
    print(f"Average Mamba pathway contribution: {np.mean(mamba_contributions):.4f}")
    print(f"Average Logistic pathway contribution: {np.mean(logistic_contributions):.4f}")
    print(f"Mamba/Logistic ratio: {np.mean(mamba_contributions)/np.mean(logistic_contributions):.2f}")

# Integrated analysis function
def comprehensive_shap_analysis(model, X_train, X_test, y_test, feature_names):
    """
    Comprehensive SHAP analysis for the hybrid Mamba model
    """
    print("Starting comprehensive SHAP analysis...")
    
    # 1. Calculate SHAP values
    shap_values, explainer = explain_mamba_model(model, X_train, X_test, feature_names, background_size=100)
    
    # 2. Summary plots
    print("\n1. Feature Importance Summary")
    plot_shap_summary(shap_values, feature_names)
    plot_shap_bar(shap_values, feature_names)
    
    # 3. Quantitative feature importance
    print("\n2. Quantitative Feature Importance")
    importance_df = plot_feature_importance(shap_values, feature_names)
    print("\nTop 10 Most Important Features:")
    for i, row in importance_df.tail(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 4. Individual prediction explanations
    print("\n3. Individual Prediction Analysis")
    analyze_individual_predictions(shap_values, X_test, feature_names)
    
    # 5. Feature dependencies
    print("\n4. Feature Dependency Analysis")
    analyze_feature_dependencies(shap_values, X_test, feature_names)
    
    # 6. Pathway analysis (for hybrid model)
    if hasattr(model, 'split_features'):
        print("\n5. Pathway Contribution Analysis")
        compare_pathway_importance(model, X_test, feature_names)
    
    # 7. Cluster analysis
    print("\n6. SHAP Value Clustering")
    plt.figure(figsize=(10, 8))
    shap.plots.heatmap(shap_values, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(output_folder + 'shap_comp_analysis.png')
    
    return shap_values, importance_df

def main():
    df_data = PWFDF_Data()
    
    X_train, y_train, scaler = df_data.prepare_data_usgs(data.all_features, split='Training')
    X_test, y_test, _ = df_data.prepare_data_usgs(data.all_features, split='Test', scaler=scaler)
    
    input_dim = X_train.shape[1]

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = n_neg / n_pos

    from models.log_reg import Staley2017Model
    from models.mamba import MambaClassifier, HybridMambaLogisticModel, ClusteredMambaModel_Flood
    from models.mp_mamba import MultiPathwayHybridModel_og
    from models.randomforest import RandomForestModel
    from models.TSMixer import TSMixerClassifier

    #model = MambaClassifier(input_dim=X_train.shape[1]).to(device)
    #model = HybridMambaLogisticModel(numerical_features, input_dim=input_dim, n_layers=1).to(device)
    #model = train_with_normalization(model, X_train, y_train, X_test, y_test)
    #model = RandomForestModel(features, random_state=None)
    #model = TSMixerClassifier(input_dim=input_dim).to(device)
    #model = MultiPathwayHybridModel_og(features=features, pos_weight=pos_weight, d_model=128, n_layers=6).to(device)
    model = ClusteredMambaModel_Flood(pos_weight, input_dim=input_dim, n_layers=1)

    model_load_path = f"/home/quinn/projects/pwfdf/output/best_models/ClusteredMamba_Flood_best.pth"
    model.load_state_dict(torch.load(model_load_path))
    
    # SHAP analysis
    print("\n" + "="*60)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    shap_values, importance_df = comprehensive_shap_analysis(model, X_train, X_test, y_test, data.all_features)
    
    # Additional insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    top_features = importance_df.tail(5)['feature'].tolist()
    print(f"Top 5 most important features: {', '.join(top_features)}")
    
    # Analyze rainfall vs non-rainfall features
    rainfall_features = ['StormAccum_mm', 'Acc015_mm', 'Acc030_mm', 'Acc060_mm']
    rainfall_importance = importance_df[importance_df['feature'].isin(rainfall_features)]['importance'].sum()
    non_rainfall_importance = importance_df[~importance_df['feature'].isin(rainfall_features)]['importance'].sum()
    
    print(f"Rainfall features total importance: {rainfall_importance:.4f} ({rainfall_importance/(rainfall_importance+non_rainfall_importance)*100:.1f}%)")
    print(f"Non-rainfall features total importance: {non_rainfall_importance:.4f} ({non_rainfall_importance/(rainfall_importance+non_rainfall_importance)*100:.1f}%)")
    
if __name__ == "__main__":
    main()