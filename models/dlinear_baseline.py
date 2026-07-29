import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    Moving average block to extract trend-cyclical components.
    """
    def __init__(self, kernel_size: int, stride: int = 1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Lookback_L, Channels)
        # Padding on both ends to keep sequence length invariant
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # Permute to (Batch, Channels, Lookback_L) for 1D pooling
        x_padded = x_padded.permute(0, 2, 1)
        out = self.avg(x_padded)
        
        # Permute back to (Batch, Lookback_L, Channels)
        out = out.permute(0, 2, 1)
        return out


class SeriesDecomp(nn.Module):
    """
    Series decomposition block into Seasonal and Trend components.
    """
    def __init__(self, kernel_size: int):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class Model(nn.Module):
    """
    DLinear Model Architecture.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = getattr(configs, 'seq_len', 96)
        self.pred_len = getattr(configs, 'pred_len', 96)

        # Decomposition kernel size
        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decompsition = SeriesDecomp(kernel_size)
        
        self.channels = getattr(configs, 'enc_in', 321)

        # Separate Linear maps mapping Temporal Lookback (L) -> Horizon (H)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        
        # Enable parameter sharing or individual weights per channel
        # Weight initialization
        self.Linear_Seasonal.weight.data.normal_(mean=0.0, std=0.1)
        self.Linear_Trend.weight.data.normal_(mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Lookback_L, Channels)
        
        # Decompose into Seasonal and Trend components
        seasonal_init, trend_init = self.decompsition(x)
        
        # Permute for linear mapping along temporal dimension: (Batch, Channels, Lookback_L)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        # Linear Forecast per component
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        # Sum components & permute back to (Batch, Horizon_H, Channels)
        x_out = seasonal_output + trend_output
        x_out = x_out.permute(0, 2, 1)
        
        return x_out  # (Batch, Pred_H, Channels)