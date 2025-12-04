import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Assuming your model and data loading code is available
# from your_model_file import ClusteredMambaModel_Flood
# from your_data_file import PWFDF_Data, all_features

class SHAPAnalyzer:
    """SHAP-based feature importance analyzer for ClusteredMambaModel_Flood"""
    
    def __init__(self, model, feature_names, device='cuda'):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained ClusteredMambaModel_Flood instance
            feature_names: List of feature names
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.model.eval()
        
    def create_model_wrapper(self):
        """
        Create a wrapper function that SHAP can use
        Returns probabilities instead of logits
        """
        def model_predict(x):
            # Convert numpy array to torch tensor
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(self.device)
            
            # Ensure correct shape: should be (batch_size, K, features)
            # where K is the number of neighbors (e.g., 6)
            # SHAP will flatten the input, so we need to reshape it
            if len(x.shape) == 2:
                # Reshape from (batch_size, K*features) to (batch_size, K, features)
                batch_size = x.shape[0]
                total_features = x.shape[1]
                # Infer K and features from total
                # You may need to adjust this based on your model's expected input
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
        Calculate SHAP values using KernelExplainer
        
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
        
        # Store original shape for later
        self.original_shape = X_background.shape  # e.g., (100, 6, 27)
        self.K = X_background.shape[1]  # Number of neighbors
        self.F = X_background.shape[2]  # Number of features
        
        # Flatten the graph structure for SHAP
        # From (batch, K, features) to (batch, K*features)
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
            link="identity"  # Use identity link since model already outputs probabilities
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
        """
        Generate SHAP summary plot
        
        Args:
            shap_values: Can be flat (samples, K*F) or reshaped (samples, K, F)
            X_explain: Original data (samples, K, F)
            aggregate_neighbors: If True, average SHAP values across neighbors
        """
        if isinstance(X_explain, torch.Tensor):
            X_explain = X_explain.cpu().numpy()
        
        # Handle different shapes
        if len(shap_values.shape) == 3:  # (samples, K, F)
            if aggregate_neighbors:
                # Average across neighbors dimension
                shap_values_plot = shap_values.mean(axis=1)  # (samples, F)
                X_explain_plot = X_explain.mean(axis=1)  # (samples, F)
                feature_names = self.feature_names
            else:
                # Flatten for plotting
                shap_values_plot = shap_values.reshape(shap_values.shape[0], -1)
                X_explain_plot = X_explain.reshape(X_explain.shape[0], -1)
                # Create feature names with neighbor indices
                feature_names = [f"{feat}_N{k}" for k in range(self.K) for feat in self.feature_names]
        else:  # Already flat
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
        """
        Generate SHAP bar plot showing mean absolute SHAP values
        
        Args:
            shap_values: Can be flat or reshaped
            aggregate_neighbors: If True, average across neighbors before plotting
        """
        if len(shap_values.shape) == 3 and aggregate_neighbors:
            # Average across neighbors
            shap_values_plot = shap_values.mean(axis=1)
            feature_names = self.feature_names
        elif len(shap_values.shape) == 3:
            # Flatten
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
    
    def plot_waterfall(self, shap_values, X_explain, sample_idx=0, 
                       save_path='./output/shap_waterfall.png'):
        """Generate waterfall plot for a single prediction"""
        if isinstance(X_explain, torch.Tensor):
            X_explain = X_explain.cpu().numpy()
        if len(X_explain.shape) == 3:
            X_explain = X_explain.squeeze(1)
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[sample_idx],
                base_values=np.mean(shap_values),
                data=X_explain[sample_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Waterfall plot saved to {save_path}")
        plt.close()
    
    def plot_force(self, explainer, shap_values, X_explain, sample_idx=0,
                   save_path='./output/shap_force.html'):
        """Generate force plot for a single prediction"""
        if isinstance(X_explain, torch.Tensor):
            X_explain = X_explain.cpu().numpy()
        if len(X_explain.shape) == 3:
            X_explain = X_explain.squeeze(1)
        
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[sample_idx],
            X_explain[sample_idx],
            feature_names=self.feature_names
        )
        shap.save_html(save_path, force_plot)
        print(f"Force plot saved to {save_path}")
    
    def get_feature_importance_ranking(self, shap_values, aggregate_neighbors=True):
        """
        Get feature importance ranking based on mean absolute SHAP values
        
        Args:
            shap_values: SHAP values (can be flat or reshaped)
            aggregate_neighbors: If True, average across neighbors
        
        Returns:
            DataFrame with features ranked by importance
        """
        if len(shap_values.shape) == 3 and aggregate_neighbors:
            # Average across neighbors first
            shap_values_agg = shap_values.mean(axis=1)  # (samples, F)
            mean_abs_shap = np.abs(shap_values_agg).mean(axis=0)
            feature_names = self.feature_names
        elif len(shap_values.shape) == 3:
            # Flatten and compute for each neighbor separately
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
    
    def analyze_feature_groups(self, shap_values):
        """
        Analyze SHAP values by feature groups (rainfall vs non-rainfall)
        """
        rainfall_features = self.model.rainfall_features
        non_rainfall_features = self.model.non_rainfall_features
        
        rainfall_indices = [i for i, f in enumerate(self.feature_names) 
                          if f in rainfall_features]
        non_rainfall_indices = [i for i, f in enumerate(self.feature_names) 
                               if f in non_rainfall_features]
        
        rainfall_importance = np.abs(shap_values[:, rainfall_indices]).mean()
        non_rainfall_importance = np.abs(shap_values[:, non_rainfall_indices]).mean()
        
        print("\n=== Feature Group Analysis ===")
        print(f"Rainfall features importance: {rainfall_importance:.4f}")
        print(f"Non-rainfall features importance: {non_rainfall_importance:.4f}")
        
        return {
            'rainfall': rainfall_importance,
            'non_rainfall': non_rainfall_importance
        }
    
import data

def main():
    """
    Main function to run SHAP analysis
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path('./output/shap_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load your data (you'll need to adapt this to your data loading code)
    print("Loading data...")
    df_data = data.PWFDF_Data()
    X_train, y_train, scaler = df_data.prepare_data_usgs(data.all_features, split='Training')
    X_test, y_test, _ = df_data.prepare_data_usgs(data.all_features, split='Test', scaler=scaler)
    
    # For this example, assuming you have loaded data:
    X_train = torch.Tensor(X_train).to(device)
    X_test = torch.Tensor(X_test).to(device)
    
    # Load trained model
    print("Loading model...")
    input_dim = X_train.shape[-1]
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = n_neg / n_pos
    
    from models.mamba import ClusteredMambaModel_Flood

    model = ClusteredMambaModel_Flood(pos_weight=pos_weight, input_dim=input_dim, n_layers=1).to(device)
    
    # Load best model weights
    model.load_state_dict(torch.load('./output/best_models/ClusteredMamba_Flood_best.pth'))
    model.eval()
    
    # Initialize SHAP analyzer
    #feature_names = data.all_features  # Your feature names list
    feature_names = model.feature_names
    analyzer = SHAPAnalyzer(model, feature_names, device)
    
    # Use a sample of training data as background (for efficiency)
    background_size = min(100, len(X_train))
    X_background = X_train[:background_size]
    
    # Use test set for explanation (or a sample if too large)
    explain_size = min(200, len(X_test))
    X_explain = X_test[:explain_size]
    
    # Calculate SHAP values
    explainer, shap_values_flat, shap_values_reshaped = analyzer.calculate_shap_values(X_background, X_explain, max_evals=100)
    
    # Generate visualizations (using reshaped values with neighbor aggregation)
    print("\nGenerating visualizations...")
    analyzer.plot_summary(shap_values_reshaped, X_explain, save_path=output_dir / 'shap_summary_aggregated.png', aggregate_neighbors=True)
    analyzer.plot_bar(shap_values_reshaped, save_path=output_dir / 'shap_bar_aggregated.png', aggregate_neighbors=True)
    
    # Also create plots without aggregation to see per-neighbor importance
    analyzer.plot_summary(shap_values_reshaped, X_explain, save_path=output_dir / 'shap_summary_per_neighbor.png', aggregate_neighbors=False)
    
    # Get feature importance ranking (aggregated across neighbors)
    importance_df = analyzer.get_feature_importance_ranking(shap_values_reshaped, aggregate_neighbors=True)
    print("\n=== Feature Importance Ranking (Aggregated) ===")
    print(importance_df.to_string(index=False))
    importance_df.to_csv(output_dir / 'feature_importance_aggregated.csv', index=False)
    
    # Also get per-neighbor importance
    importance_df_detailed = analyzer.get_feature_importance_ranking(shap_values_reshaped, aggregate_neighbors=False)
    importance_df_detailed.to_csv(output_dir / 'feature_importance_per_neighbor.csv', index=False)
    
    # Analyze feature groups
    # group_importance = analyzer.analyze_feature_groups(shap_values)
    
    print("\nSHAP analysis complete!")


if __name__ == "__main__":
    main()