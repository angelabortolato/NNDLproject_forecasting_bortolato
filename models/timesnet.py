import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


def FFT_for_Period(x: torch.Tensor, k: int = 2):
    """
    Optimized FFT period discovery without GPU-to-CPU transfer bottlenecks.
    """
    # x shape: (Batch, Sequence_Length, d_model)
    xf = torch.fft.rfft(x, dim=1)
    
    # Calculate frequency amplitudes averaged across batch and channels
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # Ignore DC component
    
    k = min(k, frequency_list.shape[0])
    _, top_list = torch.topk(frequency_list, k)
    
    seq_len = x.shape[1]
    # Keep tensor index calculations on native device
    period = [max(1, int(seq_len / freq.item())) for freq in top_list]
    
    return period, frequency_list[top_list]


class Inception_Block_2D(nn.Module):
    """
    Streamlined 2D Inception Convolutional Block (Fast multi-scale execution).
    """
    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 3):
        super(Inception_Block_2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        
        kernels = []
        for i in range(num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * i + 1,
                    padding=i
                )
            )
        self.kernels = nn.ModuleList(kernels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_list = [kernel(x) for kernel in self.kernels]
        res = torch.stack(res_list, dim=-1).mean(-1)
        return self.dropout(res)


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = getattr(configs, 'seq_len', 96)
        self.pred_len = getattr(configs, 'pred_len', 96)
        self.k = getattr(configs, 'top_k', 3)  
        self.d_model = getattr(configs, 'd_model', 32) 
        
        num_kernels = getattr(configs, 'num_kernels', 3)
        self.conv = nn.Sequential(
            Inception_Block_2D(self.d_model, self.d_model, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_2D(self.d_model, self.d_model, num_kernels=num_kernels)
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([B, (length - T), N], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
                
            out = out.reshape(B, length // period, period, N).permute(0, 3, 2, 1).contiguous()
            out = self.conv(out)
            out = out.permute(0, 3, 2, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=-1)
        period_weight = period_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, T, N, 1)
        
        res = torch.sum(res * period_weight, dim=-1)
        return self.layer_norm(res + x)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / self.affine_weight
            x = x * self.stdev + self.mean
            return x

       
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_revin = getattr(configs, 'use_revin', True)
        self.seq_len = getattr(configs, 'seq_len', 96)
        self.pred_len = getattr(configs, 'pred_len', 96)
        self.e_layers = getattr(configs, 'e_layers', 2) 
        self.enc_in = getattr(configs, 'enc_in', 321)
        self.d_model = getattr(configs, 'd_model', 32)  # 32 or 64
        self.top_k = getattr(configs, 'top_k', 3)
        if self.use_revin:
            self.revin = RevIN(self.enc_in)


        self.enc_embedding = nn.Linear(self.enc_in, self.d_model)
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(self.e_layers)])
        self.projection = nn.Linear(self.d_model, self.enc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input window instance-by-instance
        if self.use_revin:
            x = self.revin(x, 'norm')

        # Zero-pad the prediction horizon [L : L+H]
        B, L, C = x.shape
        zeros = torch.zeros([B, self.pred_len, C], device=x.device)
        x_init = torch.cat([x, zeros], dim=1)  # Shape: (B, L+H, C)
        enc_out = self.enc_embedding(x_init)
        
        for i in range(self.e_layers):
            enc_out = self.model[i](enc_out)
            
        dec_out = self.projection(enc_out)[:, -self.pred_len:, :]
        
        # Denormalize output prediction back to original scale
        if self.use_revin:
            dec_out = self.revin(dec_out, 'denorm')
        return dec_out


class ProbabilisticTimesNet(nn.Module):
    """
    Probabilistic extension of TimesNet predicting Gaussian parameters:
    Mean (mu) and Standard Deviation (sigma).
    """
    def __init__(self, args, min_sigma=1e-3):
        super(ProbabilisticTimesNet, self).__init__()
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.min_sigma = min_sigma
        
        # Base TimesNet architecture
        self.base_model = Model(args)
        
        # Separate projection heads for Mean and Std Dev
        self.head_mu = nn.Linear(args.enc_in, args.enc_in)
        self.head_sigma = nn.Linear(args.enc_in, args.enc_in)

    def forward(self, x):
        out = self.base_model(x)
        
        # Mean prediction
        mu = self.head_mu(out)
        
        # Std Dev prediction (Softplus enforces positivity + min_sigma numerical stability)
        sigma = F.softplus(self.head_sigma(out)) + self.min_sigma
        
        return mu, sigma