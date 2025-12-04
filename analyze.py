import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

class SHAPAnalyzer:
    """SHAP-based feature importance analyzer for ClusteredMambaModel_Flood and RandomForestModel"""
    
    def __init__(self, model, feature_names, device='cuda', model_type='mamba'):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained model instance (ClusteredMambaModel_Flood or RandomForestModel)
            feature_names: List of feature names (only used for Mamba, RF uses model.feature_names)
            device: 'cuda' or 'cpu'
            model_type: 'mamba' or 'random_forest'
        """
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.model_type = model_type
        self.model.eval()
        
        # For RandomForest, get the actual feature names from the model
        if model_type == 'random_forest':
            self.rf_feature_names = model.feature_names
        else:
            self.rf_feature_names = None
        
    def create_model_wrapper(self):
        """
        Create a wrapper function that SHAP can use
        Returns probabilities instead of logits
        """
        if self.model_type == 'random_forest':
            def model_predict(x):
                """Wrapper for RandomForest that extracts the limited features"""
                # Convert numpy array to torch tensor if needed
                if isinstance(x, np.ndarray):
                    x_tensor = torch.FloatTensor(x).to(self.device)
                else:
                    x_tensor = x
                
                # Ensure correct shape for RandomForest
                if len(x_tensor.shape) == 2:
                    # Add neighbor dimension: (batch_size, features) -> (batch_size, 1, features)
                    x_tensor = x_tensor.unsqueeze(1)
                
                with torch.no_grad():
                    probs, _ = self.model(x_tensor)
                
                # Return as numpy array
                return probs.cpu().numpy()
        else:
            def model_predict(x):
                """Wrapper for Mamba model"""
                # Convert numpy array to torch tensor
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x).to(self.device)
                
                # Ensure correct shape: should be (batch_size, K, features)
                if len(x.shape) == 2:
                    batch_size = x.shape[0]
                    total_features = x.shape[1]
                    K = 6  # Number of neighbors
                    features = total_features // K
                    x = x.reshape(batch_size, K, features)
                
                with torch.no_grad():
                    probs, _ = self.model(x)
                
                # Return as numpy array
                return probs.cpu().numpy()
        
        return model_predict
    
    def calculate_shap_values(self, X_background, X_explain, max_evals=100):
        """
        Calculate SHAP values using KernelExplainer or TreeExplainer (for RandomForest)
        
        Args:
            X_background: Background dataset for SHAP (e.g., training set sample)
            X_explain: Dataset to explain (e.g., test set)
            max_evals: Maximum evaluations for KernelExplainer
            
        Returns:
            explainer: SHAP explainer object
            shap_values: SHAP values for X_explain
        """
        # Convert to numpy if torch tensors
        if isinstance(X_background, torch.Tensor):
            X_background = X_background.cpu().numpy()
        if isinstance(X_explain, torch.Tensor):
            X_explain = X_explain.cpu().numpy()
        
        if self.model_type == 'random_forest':
            # For RandomForest, use TreeExplainer which is much faster and exact
            print("Using TreeExplainer for RandomForest...")
            
            # Extract the limited features that the RF model uses
            X_background_limited = X_background[:, 0, self.model.feature_indices]
            X_explain_limited = X_explain[:, 0, self.model.feature_indices]
            
            # Create TreeExplainer (much faster than KernelExplainer)
            explainer = shap.TreeExplainer(self.model.rf)
            
            # Calculate SHAP values
            print(f"Calculating SHAP values for {len(X_explain_limited)} samples...")
            shap_values = explainer.shap_values(X_explain_limited)
            
            # For binary classification, shap_values is a list [class0, class1] or array with shape (..., 2)
            # We want the positive class (class 1)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3 and shap_values.shape[-1] == 2:
                # Shape is (samples, features, 2) - take positive class
                shap_values = shap_values[:, :, 1]
            
            print(f"Final SHAP values shape: {shap_values.shape}")
            
            # Store info for plotting
            self.X_explain_limited = X_explain_limited
            
            return explainer, shap_values, shap_values
        
        else:
            # For Mamba model, use KernelExplainer
            # Store original shape for later
            self.original_shape = X_background.shape  # e.g., (100, 6, 27)
            self.K = X_background.shape[1]  # Number of neighbors
            self.F = X_background.shape[2]  # Number of features
            
            # Flatten the graph structure for SHAP
            X_background_flat = X_background.reshape(X_background.shape[0], -1)
            X_explain_flat = X_explain.reshape(X_explain.shape[0], -1)
            
            print(f"Flattened shapes: Background {X_background_flat.shape}, Explain {X_explain_flat.shape}")
            
            # Create model wrapper
            model_fn = self.create_model_wrapper()
            
            # Initialize SHAP KernelExplainer
            print("Initializing SHAP explainer...")
            explainer = shap.KernelExplainer(
                model_fn, 
                X_background_flat,
                link="identity"
            )
            
            # Calculate SHAP values
            print(f"Calculating SHAP values for {len(X_explain_flat)} samples...")
            shap_values = explainer.shap_values(
                X_explain_flat,
                nsamples=max_evals,
                silent=False
            )
            
            # Reshape SHAP values back to (samples, K, features)
            shap_values_reshaped = shap_values.reshape(-1, self.K, self.F)
            
            return explainer, shap_values, shap_values_reshaped
    
    def plot_summary(self, shap_values, X_explain, save_path='./output/shap_summary.png', 
                     aggregate_neighbors=True):
        """Generate SHAP summary plot"""
        if self.model_type == 'random_forest':
            # For RF, use the limited features
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values, 
                self.X_explain_limited, 
                feature_names=self.rf_feature_names,
                show=False
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to {save_path}")
            plt.close()
            return
        
        # For Mamba model
        if isinstance(X_explain, torch.Tensor):
            X_explain = X_explain.cpu().numpy()
        
        if len(shap_values.shape) == 3:
            if aggregate_neighbors:
                shap_values_plot = shap_values.mean(axis=1)
                X_explain_plot = X_explain.mean(axis=1)
                feature_names = self.feature_names
            else:
                shap_values_plot = shap_values.reshape(shap_values.shape[0], -1)
                X_explain_plot = X_explain.reshape(X_explain.shape[0], -1)
                feature_names = [f"{feat}_N{k}" for k in range(self.K) for feat in self.feature_names]
        else:
            shap_values_plot = shap_values
            X_explain_plot = X_explain.reshape(X_explain.shape[0], -1)
            feature_names = [f"{feat}_N{k}" for k in range(self.K) for feat in self.feature_names]
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_plot, 
            X_explain_plot, 
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to {save_path}")
        plt.close()
    
    def plot_bar(self, shap_values, save_path='./output/shap_bar.png', aggregate_neighbors=True):
        """Generate SHAP bar plot showing mean absolute SHAP values"""
        if self.model_type == 'random_forest':
            # For RF, use the limited features
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                feature_names=self.rf_feature_names,
                plot_type="bar",
                show=False
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Bar plot saved to {save_path}")
            plt.close()
            return
        
        # For Mamba model
        if len(shap_values.shape) == 3 and aggregate_neighbors:
            shap_values_plot = shap_values.mean(axis=1)
            feature_names = self.feature_names
        elif len(shap_values.shape) == 3:
            shap_values_plot = shap_values.reshape(shap_values.shape[0], -1)
            feature_names = [f"{feat}_N{k}" for k in range(self.K) for feat in self.feature_names]
        else:
            shap_values_plot = shap_values
            feature_names = [f"{feat}_N{k}" for k in range(self.K) for feat in self.feature_names]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_plot,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar plot saved to {save_path}")
        plt.close()
    
    def get_feature_importance_ranking(self, shap_values, aggregate_neighbors=True):
        """
        Get feature importance ranking based on mean absolute SHAP values
        
        Returns:
            DataFrame with features ranked by importance
        """
        if self.model_type == 'random_forest':
            # For RF, use the limited features
            print(f"DEBUG: shap_values shape: {shap_values.shape}")
            print(f"DEBUG: Number of RF features: {len(self.rf_feature_names)}")
            
            # SHAP values should be (samples, features)
            if len(shap_values.shape) == 2:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
            else:
                raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
            
            print(f"DEBUG: mean_abs_shap shape: {mean_abs_shap.shape}")
            
            # Verify lengths match
            if len(mean_abs_shap) != len(self.rf_feature_names):
                raise ValueError(f"Mismatch: {len(mean_abs_shap)} SHAP values but {len(self.rf_feature_names)} feature names")
            
            feature_names = self.rf_feature_names
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean_Abs_SHAP': mean_abs_shap
            })
            importance_df = importance_df.sort_values('Mean_Abs_SHAP', ascending=False)
            importance_df['Rank'] = range(1, len(importance_df) + 1)
            
            return importance_df
        
        # For Mamba model
        if len(shap_values.shape) == 3 and aggregate_neighbors:
            shap_values_agg = shap_values.mean(axis=1)
            mean_abs_shap = np.abs(shap_values_agg).mean(axis=0)
            feature_names = self.feature_names
        elif len(shap_values.shape) == 3:
            shap_values_flat = shap_values.reshape(shap_values.shape[0], -1)
            mean_abs_shap = np.abs(shap_values_flat).mean(axis=0)
            feature_names = [f"{feat}_N{k}" for k in range(self.K) for feat in self.feature_names]
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_names = [f"{feat}_N{k}" for k in range(self.K) for feat in self.feature_names]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Abs_SHAP': mean_abs_shap
        })
        importance_df = importance_df.sort_values('Mean_Abs_SHAP', ascending=False)
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        
        return importance_df


import data

def main():
    """
    Main function to run SHAP analysis for both Mamba and RandomForest models
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path('./output/shap_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df_data = data.PWFDF_Data()
    X_train, y_train, scaler = df_data.prepare_data_usgs(data.all_features, split='Training')
    X_test, y_test, _ = df_data.prepare_data_usgs(data.all_features, split='Test', scaler=scaler)
    
    X_train = torch.Tensor(X_train).to(device)
    X_test = torch.Tensor(X_test).to(device)
    
    # ==================== MAMBA MODEL ANALYSIS ====================
    print("\n" + "="*60)
    print("ANALYZING MAMBA MODEL")
    print("="*60)
    
    input_dim = X_train.shape[-1]
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = n_neg / n_pos
    
    from models.mamba import ClusteredMambaModel_Flood
    
    mamba_model = ClusteredMambaModel_Flood(pos_weight=pos_weight, input_dim=input_dim, n_layers=1).to(device)
    mamba_model.load_state_dict(torch.load('./output/best_models/ClusteredMamba_Flood_best.pth'))
    mamba_model.eval()
    
    # Initialize SHAP analyzer for Mamba
    mamba_analyzer = SHAPAnalyzer(mamba_model, data.all_features, device, model_type='mamba')
    
    # Use samples for efficiency
    background_size = min(100, len(X_train))
    explain_size = min(200, len(X_test))
    X_background = X_train[:background_size]
    X_explain = X_test[:explain_size]
    
    # Calculate SHAP values for Mamba
    explainer, shap_values_flat, shap_values_reshaped = mamba_analyzer.calculate_shap_values(
        X_background, X_explain, max_evals=100
    )
    
    # Generate Mamba visualizations
    print("\nGenerating Mamba visualizations...")
    mamba_analyzer.plot_summary(
        shap_values_reshaped, X_explain, 
        save_path=output_dir / 'mamba_shap_summary_aggregated.png', 
        aggregate_neighbors=True
    )
    mamba_analyzer.plot_bar(
        shap_values_reshaped, 
        save_path=output_dir / 'mamba_shap_bar_aggregated.png', 
        aggregate_neighbors=True
    )
    
    # Get Mamba feature importance
    mamba_importance_df = mamba_analyzer.get_feature_importance_ranking(
        shap_values_reshaped, aggregate_neighbors=True
    )
    print("\n=== Mamba Feature Importance Ranking ===")
    print(mamba_importance_df.to_string(index=False))
    mamba_importance_df.to_csv(output_dir / 'mamba_feature_importance.csv', index=False)
    
    # ==================== RANDOM FOREST ANALYSIS ====================
    print("\n" + "="*60)
    print("ANALYZING RANDOM FOREST MODEL")
    print("="*60)
    
    from models.randomforest import RandomForestModel, train_random_forest
    
    # Initialize and train RandomForest
    rf_model = RandomForestModel(n_estimators=100, random_state=42).to(device)
    rf_model = train_random_forest(rf_model, (X_train, y_train, X_test, y_test))
    
    # Initialize SHAP analyzer for RandomForest
    rf_analyzer = SHAPAnalyzer(rf_model, data.all_features, device, model_type='random_forest')
    
    # Calculate SHAP values for RandomForest (TreeExplainer is much faster)
    rf_explainer, rf_shap_values, _ = rf_analyzer.calculate_shap_values(
        X_background, X_explain
    )
    
    # Generate RandomForest visualizations
    print("\nGenerating RandomForest visualizations...")
    rf_analyzer.plot_summary(
        rf_shap_values, X_explain,
        save_path=output_dir / 'rf_shap_summary.png'
    )
    rf_analyzer.plot_bar(
        rf_shap_values,
        save_path=output_dir / 'rf_shap_bar.png'
    )
    
    # Get RandomForest feature importance
    rf_importance_df = rf_analyzer.get_feature_importance_ranking(rf_shap_values)
    print("\n=== RandomForest Feature Importance Ranking ===")
    print(rf_importance_df.to_string(index=False))
    rf_importance_df.to_csv(output_dir / 'rf_feature_importance.csv', index=False)
    
    # Print comparison
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE COMPARISON")
    print("="*60)
    print("\nTop 5 features for each model:")
    print("\nMamba Model:")
    print(mamba_importance_df.head(5).to_string(index=False))
    print("\nRandomForest Model:")
    print(rf_importance_df.head(5).to_string(index=False))
    
    print("\nSHAP analysis complete!")


if __name__ == "__main__":
    main()