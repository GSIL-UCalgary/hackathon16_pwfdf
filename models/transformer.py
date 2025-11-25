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