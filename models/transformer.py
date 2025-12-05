import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 256
n_embd = 128
n_head = 4
n_layer = 6
dropout = 0.2
head_size = n_embd // n_head

class Head(nn.Module):
  """ one head of self-attention """

  def __init__(self):
      super().__init__()
      self.key = nn.Linear(n_embd, head_size, bias=False)
      self.query = nn.Linear(n_embd, head_size, bias=False)
      self.value = nn.Linear(n_embd, head_size, bias=False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

      self.dropout = nn.Dropout(dropout)

  def forward(self, x):
      # input of size (batch, time-step, channels)
      # output of size (batch, time-step, head size)
      B,T,C = x.shape
      k = self.key(x)   # (B,T,hs)
      q = self.query(x) # (B,T,hs)
      # compute attention scores ("affinities")
      wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
      #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
      wei = F.softmax(wei, dim=-1) # (B, T, T)
      wei = self.dropout(wei)
      # perform the weighted aggregation of the values
      v = self.value(x) # (B,T,hs)
      out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
      return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedFoward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    #self.token_embedding_table = nn.Embedding(app.vocab_size, app.n_embd)
    #self.position_embedding_table = nn.Embedding(app.block_size, app.n_embd)

    self.input_projection = torch.nn.Linear(input_size, n_embd)
    self.pos_encoding = torch.nn.Parameter(torch.randn(1, 1, n_embd))

    self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, input_size)

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
          torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B,T) tensor of integers
    #tok_emb = self.token_embedding_table(idx) # (B,T,C)
    #x = tok_emb + pos_emb # (B,T,C)

    x = idx.unsqueeze(1)
    x = self.input_projection(x)
    x = x + self.pos_encoding

    x = self.blocks(x) # (B,T,C)
    x = self.ln_f(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes=1, n_embd=128, n_head=4, n_layer=4, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.duration = '15min'
        
        # Input projection
        self.input_proj = nn.Linear(input_size, n_embd)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4*n_embd,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(n_embd, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        x = self.input_proj(x)  # (batch_size, 1, n_embd)
        x = self.transformer(x)  # (batch_size, 1, n_embd)
        x = x.squeeze(1)  # (batch_size, n_embd)
        return self.classifier(x).squeeze(-1)
    
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Optional, Tuple

from data import all_features

class SimpleTransformerClassifier(nn.Module):
    """
    Corrected Transformer-based model for binary classification.

    Correction: Treats KF (Soil Erodibility Index) as a continuous numerical feature.
    All 4 selected features are now treated as numerical "tokens" in the sequence.

    1. All 4 features are embedded using nn.Linear.
    2. Uses a learnable [CLS] token for dedicated sequence aggregation.
    """
    
    def __init__(self, duration='15min', d_model=64, nhead=4, dim_feedforward=128, num_layers=1, dropout=0.1, **kwargs):
        """
        Initializes the Transformer classifier.
        
        Args:
            duration (str): Used for selecting the correct rainfall feature.
            d_model (int): The dimension of the model (input/output of attention).
        """
        super().__init__()
        self.name = 'TFClassifier'
        self.d_model = d_model
        
        # 1. Feature Selection Logic (Matches RandomForestModel)
        # ALL 4 features are now treated as Numerical/Continuous
        self.numerical_features = ['PropHM23', 'dNBR/1000', 'KF', 'Acc015_mm']
        
        # Get indices for feature selection from the full list
        self.feature_indices = [all_features.index(feat) for feat in self.numerical_features if feat in all_features]
        self.num_features = len(self.feature_indices) # This is 4
        
        # The sequence length is the number of features + CLS token
        self.sequence_length = self.num_features + 1 # 4 features + 1 for [CLS]
        
        # 2. Feature Embeddings (All 4 features are numerical)
        # A. Embedding for ALL Numerical Features (4 tokens)
        # We map the 1D numerical value to the d_model dimension
        self.numerical_embedding = nn.Linear(1, d_model)

        # B. [CLS] Token (The 0th token in the sequence)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 3. Positional Encoding (Learnable, size depends on the new sequence length)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.sequence_length, d_model))
        
        # 4. Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Classification Head (MLP)
        # This processes the [CLS] token's output embedding
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1) # Output a single logit for binary classification
        )

        # 6. Loss Function
        self.loss_fn = nn.BCEWithLogitsLoss()


    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict logits and calculate loss (if target is provided).
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, num_total_features).
            target (torch.Tensor, optional): Labels (B,).
        
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Predicted probabilities (after sigmoid), shape (B,).
                - Calculated loss, or None if target is not provided.
        """
        B = x.size(0)
        
        # 1. Feature Selection
        # Select all 4 numerical features (B, 4)
        numerical_data = x[:, 0, self.feature_indices] 
        
        # 2. Feature Embedding
        # Reshape to (B, 4, 1) and map to d_model (B, 4, d_model)
        feature_sequence = self.numerical_embedding(numerical_data.unsqueeze(-1))
        
        # 3. Concatenate [CLS] Token
        # cls_tokens (B, 1, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        
        # enc_input (B, 5, d_model): ([CLS], Feature_1, Feature_2, Feature_3, Feature_4)
        enc_input = torch.cat([cls_tokens, feature_sequence], dim=1)

        # 4. Add Positional Encoding
        enc_input = enc_input + self.positional_encoding.expand(B, -1, -1)
        
        # 5. Transformer Encoder Pass
        # transformer_output (B, 5, d_model)
        transformer_output = self.transformer_encoder(enc_input)
        
        # 6. Extract [CLS] Token Output
        # cls_output (B, d_model)
        cls_output = transformer_output[:, 0, :]
        
        # 7. Classification Head
        # Logits shape: (B, 1) -> squeezed to (B,)
        logits = self.classification_head(cls_output).squeeze(-1)
        
        # 8. Compute Probabilities and Loss
        probs = torch.sigmoid(logits)
        
        loss = None
        if target is not None:
            loss = self.loss_fn(logits, target.float())

        return probs, loss
    
from eval import ComboLoss

class AttentionClassifier(nn.Module):
    def __init__(self, features, d_model=128, n_heads=8, n_layers=3, dropout=0.3, 
                 ff_dim=256):
        super().__init__()
        self.feature_indices = [all_features.index(feat) for feat in features if feat in all_features]
        self.hyperparameters = {
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dropout': dropout,
            'ff_dim': ff_dim
        }
        self.input_dim = len(self.feature_indices)
        self.d_model = d_model
        self.name = 'Attention'
        
        # Project each feature to d_model
        self.feature_projection = nn.Linear(1, d_model)
        
        # Learnable feature embeddings (positional encoding for features)
        self.feature_embeddings = nn.Parameter(
            torch.randn(1, self.input_dim, d_model) * 0.02
        )
        
        # Multi-head self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm is more stable
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )
        
        self.loss_fn = ComboLoss()
        
    def forward(self, x, target=None):
        # x: B, T, F -> take first timestep
        x = x[:, 0, self.feature_indices]  # B, F
        batch_size = x.shape[0]
        
        # Project each feature independently
        x = x.unsqueeze(-1)  # B, F, 1
        x = self.feature_projection(x)  # B, F, d_model
        
        # Add feature embeddings (like positional encoding)
        x = x + self.feature_embeddings
        
        # Self-attention across features
        x = self.transformer(x)  # B, F, d_model
        
        # Aggregate: mean pooling over features
        x = x.mean(dim=1)  # B, d_model
        
        # Classification
        x = self.output_head(x).squeeze(-1)
        probs = torch.sigmoid(x)
        
        if target is not None:
            loss = self.loss_fn(x, target)
            return probs, loss
        return probs, None