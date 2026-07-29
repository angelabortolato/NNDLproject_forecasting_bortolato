import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    """
    Standard 1D Convolutional Block with Residual Connection.
    """
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super(Conv1DBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels  # Depthwise convolution across temporal dimension
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Channels, Lookback_L)
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return out + residual  


class Model(nn.Module):
    """
    1D-ConvNet Baseline Model for LSTF.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = getattr(configs, 'seq_len', 96)
        self.pred_len = getattr(configs, 'pred_len', 96)
        self.channels = getattr(configs, 'enc_in', 321)
        
        kernel_size = getattr(configs, 'kernel_size', 3)
        num_layers = getattr(configs, 'num_layers', 2)
        
        # Stacked 1D Convolutional Layers
        self.conv_layers = nn.ModuleList([
            Conv1DBlock(channels=self.channels, kernel_size=kernel_size)
            for _ in range(num_layers)
        ])
        
        # Temporal projection: Maps lookback steps (L) to prediction steps (H)
        self.projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Lookback_L, Channels)
        
        # Permute for 1D PyTorch Conv: (Batch, Channels, Lookback_L)
        x = x.permute(0, 2, 1)
        
        # Apply 1D Convolutional Blocks
        for conv in self.conv_layers:
            x = conv(x)
            
        # Project along temporal dimension L -> H: (Batch, Channels, Pred_H)
        x = self.projection(x)
        
        # Permute back to standard output shape: (Batch, Pred_H, Channels)
        out = x.permute(0, 2, 1)
        return out