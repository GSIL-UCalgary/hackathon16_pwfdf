import torch
import torch.nn as nn
import torch.nn.functional as F

# Use BCEWithLogitsLoss for numerical stability, which combines sigmoid and BCELoss.
# The forward pass will return the output of this loss function.
class FixedNeighborhoodGNN(nn.Module):
    """
    A Graph Neural Network model designed for a fixed, local neighborhood.

    Input (x): [B, N, F], where B=Batch, N=4 (0=Main Node, 1-3=Neighbors), F=Features.
    Target (target): [B, 1], Binary labels for the main node (N=0).
    """
    def __init__(self, input_features: int, hidden_features: int, output_features: int = 1):
        super().__init__()
        self.name = 'FixedNGNN'

        # Define the weight matrix for the main node (N=0) features
        self.main_node_transform = nn.Linear(input_features, hidden_features)

        # Define the weight matrix for the aggregated neighbor features (N=1, 2, 3)
        # The input size is 'input_features' because we aggregate (e.g., mean) first.
        self.neighbor_agg_transform = nn.Linear(input_features, hidden_features)

        # Final prediction layer: Takes combined features (2 * hidden_features) and outputs 1 logit.
        self.output_layer = nn.Linear(hidden_features * 2, output_features)

        # Loss function: BCEWithLogitsLoss is used for numerical stability
        # It internally applies sigmoid to the logits before calculating the binary cross-entropy.
        self.loss_fn = nn.BCELoss()
        self.seg = nn.Sigmoid()

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        # 1. Separate Main Node and Neighbors
        # x_main: [B, F] - Features of the central node (N=0)
        x_main = x[:, 0, :]
        # x_neighbors: [B, 3, F] - Features of the three neighbors (N=1, 2, 3)
        x_neighbors = x[:, 1:, :]

        # 2. Message Passing and Aggregation (Simple Mean Aggregation)
        # Aggregate features from the 3 neighbors by taking the mean across the neighbor dimension (dim=1).
        # x_agg: [B, F]
        x_agg = torch.mean(x_neighbors, dim=1)

        # 3. Transformation (Applying Linear Layers and Activation)
        # Transform main node features
        h_main = F.relu(self.main_node_transform(x_main)) # [B, H]
        # Transform aggregated neighbor features
        h_agg = F.relu(self.neighbor_agg_transform(x_agg)) # [B, H]

        # 4. Combination
        # Concatenate the transformed main node and aggregated neighbor features
        h_combined = torch.cat([h_main, h_agg], dim=1) # [B, 2*H]

        # 5. Prediction (Logits)
        # Apply final linear layer to get the raw output logits (before sigmoid)
        logits = self.output_layer(h_combined) # [B, 1]
        probs = self.seg(logits)

        # 6. Calculate Loss
        # The target must be float and logits must be the raw output.
        if target is not None:
            loss = self.loss_fn(probs.squeeze(1), target.float())
            return probs, loss

        return probs, None

# --- Example Usage ---
if __name__ == '__main__':
    # Configuration
    batch_size = 32
    input_features = 10
    hidden_features = 16
    num_nodes = 4 # Main node (0) + 3 Neighbors (1, 2, 3)

    # 1. Initialize Model
    model = FixedNeighborhoodGNN(input_features, hidden_features)
    print(f"Model initialized with {input_features} input features and {hidden_features} hidden features.")
    print(model)

    # 2. Generate Mock Data
    # x: [B, N, F] -> [32, 4, 10]
    # N=0 is the main node. N=1,2,3 are neighbors.
    x_input = torch.randn(batch_size, num_nodes, input_features)

    # target: [B, 1] -> [32, 1]. Binary labels (0 or 1).
    target_labels = torch.randint(0, 2, (batch_size, 1)).float()

    print(f"\nInput shape (x): {x_input.shape}")
    print(f"Target shape: {target_labels.shape}")

    # 3. Perform Forward Pass
    try:
        # The forward method returns (loss, logits)
        loss_value, raw_logits = model.forward(x_input, target_labels)
        
        # Calculate predicted probabilities (by applying sigmoid to the logits)
        probabilities = torch.sigmoid(raw_logits)
        
        # Calculate hard predictions (0 or 1)
        predictions = (probabilities > 0.5).int()

        print("\n--- Model Output ---")
        print(f"Calculated Loss (BCELoss with Logits): {loss_value.item():.4f}")
        print(f"Output Logits shape: {raw_logits.shape}")
        print(f"Sample Predicted Probabilities (first 5): {probabilities[:5].flatten().tolist()}")
        print(f"Sample Hard Predictions (first 5): {predictions[:5].flatten().tolist()}")
        print(f"Sample True Targets (first 5): {target_labels[:5].flatten().tolist()}")

    except Exception as e:
        print(f"\nAn error occurred during the forward pass: {e}")

    # 4. Example of a backward pass (training step)
    # loss_value.backward()
    # optimizer.step() # (Requires an optimizer)