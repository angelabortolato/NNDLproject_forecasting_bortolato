import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Improved LSTM Baseline with RevIN Normalization and Channel-Wise Projections.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = getattr(configs, 'seq_len', 96)
        self.pred_len = getattr(configs, 'pred_len', 96)
        self.channels = getattr(configs, 'enc_in', 321)
        
        hidden_dim = getattr(configs, 'hidden_dim', 32) # Reduced to 32 for MPS stability
        num_layers = getattr(configs, 'num_layers', 1)
        
        # RevIN Normalization Parameters
        self.affine_weight = nn.Parameter(torch.ones(1, 1, self.channels))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.channels))

        # LSTM processes sequence
        self.lstm = nn.LSTM(
            input_size=self.channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 3. Channel-Independent Linear Projection: maps (L * hidden_dim) -> H per channel
        # Size: (seq_len * hidden_dim) -> pred_len (ONLY ~300k parameters instead of 189M!)
        self.temporal_fc = nn.Linear(self.seq_len * hidden_dim, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, C)
        B, L, C = x.shape
        
        #  RevIN / Instance Normalization 
        seq_mean = x.mean(dim=1, keepdim=True)
        seq_std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - seq_mean) / seq_std
        x_norm = x_norm * self.affine_weight + self.affine_bias

        #  LSTM Pass 
        lstm_out, _ = self.lstm(x_norm) # (B, L, hidden_dim)
        
        #  Efficient Flattening & Temporal Projection 
        # Reshape to process channels independently: (B, C, L * hidden_dim)
        # We expand hidden states across channels:
        lstm_out = lstm_out.unsqueeze(2).repeat(1, 1, C, 1) # (B, L, C, hidden_dim)
        lstm_out = lstm_out.permute(0, 2, 1, 3).reshape(B, C, -1) # (B, C, L * hidden_dim)
        
        # Apply shared linear projection along time dimension: (B, C, H)
        out = self.temporal_fc(lstm_out)
        out = out.permute(0, 2, 1) # Reshape back to (B, H, C)
        
        #  Denormalization 
        out = (out - self.affine_bias) / (self.affine_weight + 1e-5)
        out = out * seq_std + seq_mean
        
        return out