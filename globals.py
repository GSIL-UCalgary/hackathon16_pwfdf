import torch

all_features = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
