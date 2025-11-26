import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, seq_len, enc_in, d_model, dropout):
        super(ResBlock, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout)
        )
        self.channel = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, enc_in),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: [B, L, D]
        # Temporal mixing (across time/sequence dimension)
        #x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        # Channel mixing (across feature dimension)
        x = x + self.channel(x)
        return x


class TSMixerClassifier(nn.Module):
    def __init__(self, input_dim=14, d_model=64, n_layers=4, dropout=0.1, seq_len=1):
        super(TSMixerClassifier, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len  # For tabular data, typically 1
        self.duration = '15min'
        self.name = 'TSMixer'
        
        # Stack of ResBlocks
        self.model = nn.ModuleList([
            ResBlock(seq_len=seq_len, enc_in=input_dim, d_model=d_model, dropout=dropout) 
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, seq_len, input_dim] -> [B, seq_len * input_dim]
            nn.Linear(seq_len * input_dim, 1),
            #nn.ReLU(),
            #nn.Dropout(dropout),
            #nn.Linear(d_model, 32),
            #nn.ReLU(),
            #nn.Dropout(dropout),
            #nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, input_dim] for tabular data
        if len(x.shape) == 2:
            # Reshape to [B, seq_len, input_dim]
            x = x.unsqueeze(1)  # [B, 1, input_dim]
        
        for block in self.model:
            x = block(x)
        
        return self.classifier(x).squeeze(-1)