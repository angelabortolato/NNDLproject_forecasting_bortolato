import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import gzip
import pandas as pd



def visualize_raw_datasets(data_dir: str = "./data", save_dir: str = "./plots"):
    """
    Loads and visualizes raw time series data for Electricity and Exchange Rate datasets.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------------
    #  Raw Electricity Data
    # ----------------------------------------------------
    elec_path = os.path.join(data_dir, "electricity.txt.gz")
    if os.path.exists(elec_path):
        with gzip.open(elec_path, 'rt') as f:
            df_elec = pd.read_csv(f, sep=r'\s+|,', header=None, engine='python')

        print(f"--> Electricity raw shape (Time Steps x Channels): {df_elec.shape}")

        fig, axes = plt.subplots(2, 1, figsize=(14, 7))

        # Full time horizon
        axes[0].plot(df_elec.iloc[:, 0], label="Client 0", alpha=0.8, linewidth=0.8)
        axes[0].plot(df_elec.iloc[:, 1], label="Client 1", alpha=0.8, linewidth=0.8)
        axes[0].plot(df_elec.iloc[:, 2], label="Client 2", alpha=0.8, linewidth=0.8)
        axes[0].set_title("Electricity Dataset - Full Time Horizon (~26,000 Hourly Steps)", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Time Step (Hours)")
        axes[0].set_ylabel("Electricity Load (kWh)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        # Zoomed-in window (2 weeks = 336 hours)
        zoomed_hours = 336
        axes[1].plot(df_elec.iloc[:zoomed_hours, 0], label="Client 0", color="tab:blue", linewidth=1.8)
        axes[1].plot(df_elec.iloc[:zoomed_hours, 1], label="Client 1", color="tab:orange", linewidth=1.8)

        # Highlight 24-hour daily periodicity lines
        for day in range(1, 14):
            axes[1].axvline(x=day * 24, color='gray', linestyle='--', alpha=0.5)

        axes[1].set_title("Electricity Dataset - Zoomed-In View (14 Days / 336 Hours) Showing Clear 24-Hour Periodicity", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Time Step (Hours)")
        axes[1].set_ylabel("Electricity Load (kWh)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        elec_save_path = os.path.join(save_dir, "raw_electricity_visualization.png")
        plt.savefig(elec_save_path, dpi=300)
        plt.show()
    else:
        print(f"--> File not found: {elec_path}. Skipping Electricity raw plot.")

    # ----------------------------------------------------
    #  Raw Exchange Rate Data
    # ----------------------------------------------------
    exch_path = os.path.join(data_dir, "exchange_rate.txt.gz")
    if os.path.exists(exch_path):
        with gzip.open(exch_path, 'rt') as f:
            df_exch = pd.read_csv(f, sep=r'\s+|,', header=None, engine='python')

        print(f"--> Exchange Rate raw shape (Time Steps x Channels): {df_exch.shape}")

        plt.figure(figsize=(14, 5))
        for c in range(min(5, df_exch.shape[1])):
            plt.plot(df_exch.iloc[:, c], label=f"Currency Pair {c}", linewidth=1.2)

        plt.title("Financial Exchange Rate Dataset - Full Time Horizon (~7,500 Daily Steps)", fontsize=12, fontweight='bold')
        plt.xlabel("Time Step (Days)")
        plt.ylabel("Exchange Rate Value")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()

        exch_save_path = os.path.join(save_dir, "raw_exchange_rate_visualization.png")
        plt.savefig(exch_save_path, dpi=300)
        plt.show()
    else:
        print(f"--> File not found: {exch_path}. Skipping Exchange Rate raw plot.")

    print(f"--> Raw data plots saved under '{save_dir}/'")


MODEL_COLORS = {
    "DLinear": "tab:blue",
    "1D-ConvNet": "tab:orange",
    "LSTM": "tab:green",
    "TimesNet": "tab:red"
}


def plot_loss_trajectories(all_results, save_path="./plots/loss_trajectories.png"):
    """
    Plots all models' training (solid) and validation (dashed) curves on a single panel.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    
    for (model_name, res) in all_results.items():
        epochs = range(1, len(res["train_losses"]) + 1)
        color = MODEL_COLORS.get(model_name, "tab:gray")
        
        # Train (solid)
        plt.plot(epochs, res["train_losses"], label=f"{model_name} (Train)", 
                 color=color, linestyle="-", linewidth=2)
        # Val (dashed)
        plt.plot(epochs, res["val_losses"], label=f"{model_name} (Val)", 
                 color=color, linestyle="--", linewidth=2)

    plt.title("Training vs. Validation Loss Trajectories Across Models", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"--> Loss trajectory plot saved to {save_path}")


def plot_forecast_comparison(models_dict, test_loader, device, sample_idx=0, channel_idx=0, 
                             seq_len=96, pred_len=96, save_path="./plots/forecast_comparison.png"):
    """
    Generates a multi-model comparative forecast plot against ground truth observations.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract a batch from test_loader
    batch_x, batch_y = next(iter(test_loader))
    batch_x = batch_x.to(device)
    
    # Ground truth values for selected sample and channel
    lookback = batch_x[sample_idx, :, channel_idx].cpu().numpy()
    target_y = batch_y[sample_idx, -pred_len:, channel_idx].cpu().numpy()
    
    full_ground_truth = np.concatenate([lookback, target_y])
    time_lookback = np.arange(seq_len)
    time_horizon = np.arange(seq_len, seq_len + pred_len)

    plt.figure(figsize=(12, 5))
    
    # Plot historical lookback and actual target sequence
    plt.plot(time_lookback, lookback, color='black', label='Historical Lookback (L=96)', linewidth=1.5)
    plt.plot(time_horizon, target_y, color='black', linestyle=':', label='Ground Truth (H=96)', linewidth=2.0)

    # Plot each model's prediction
    for (name, model) in models_dict.items():
        model.eval()
        color = MODEL_COLORS.get(name, "tab:gray")
        with torch.no_grad():
            pred = model(batch_x)[sample_idx, :, channel_idx].cpu().numpy()
            plt.plot(time_horizon, pred, label=f'{name} Forecast', color=color, linewidth=1.8, linestyle='--')

    plt.axvline(x=seq_len, color='gray', linestyle='--', alpha=0.7, label='Forecast Boundary')
    plt.title(f"Visual Forecast Comparison (Channel {channel_idx})", fontsize=13, fontweight='bold')
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Value")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"--> Forecast visual plot saved to {save_path}")

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def compute_model_residuals(model, test_loader, device, pred_len=96):
    """
    Computes ground truth (y), predictions (y_hat), and residuals (e = y - y_hat) 
    across the entire test dataset.
    """
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            target_y = batch_y[:, -pred_len:, :].cpu().numpy()
            
            # Predict
            pred_y = model(batch_x).cpu().numpy()
            
            all_targets.append(target_y)
            all_preds.append(pred_y)
            
    targets = np.concatenate(all_targets, axis=0) # Shape: (N_samples, H, C)
    preds = np.concatenate(all_preds, axis=0)     # Shape: (N_samples, H, C)
    residuals = targets - preds                    # Shape: (N_samples, H, C)
    
    return targets, preds, residuals


def plot_residual_diagnostics(models_dict, test_loader, device, 
                                         channel_idx=0, pred_len=96, 
                                         save_path="./plots/residual_diagnostics_comparison.png"):
    """
    Plots the 2 most critical residual diagnostics for ALL models in models_dict:
    1. Overlay Residual Distribution (KDE / Density) vs. Zero-Bias Normal Fit
    2. Step-by-Step Absolute Residual Error across the Forecast Horizon (t+1 to t+H)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Model color mapping (consistent with main benchmark palette)
    MODEL_COLORS = {
        "DLinear": "tab:blue",
        "1D-ConvNet": "tab:orange",
        "LSTM": "tab:green",
        "TimesNet": "tab:red"
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Collect data for all models
    for name, model in models_dict.items():
        color = MODEL_COLORS.get(name, "tab:gray")
        
        # Compute and unpack residuals
        targets, preds, residuals = compute_model_residuals(model, test_loader, device, pred_len=pred_len)
        
        # Isolate selected channel
        res_chan = residuals[:, :, channel_idx]
        res_flat = res_chan.flatten()
        targets_flat = targets[:, :, channel_idx].flatten()
        
        # ----------------------------------------------------
        # Panel 1: Residual Density Plot (KDE Fit)
        # ----------------------------------------------------
        kde = stats.gaussian_kde(res_flat)
        x_grid = np.linspace(res_flat.min(), res_flat.max(), 200)
        mean_err = np.mean(res_flat)
        std_err = np.std(res_flat)
        
        ax1.plot(x_grid, kde(x_grid), color=color, linewidth=2.2, 
                 label=f"{name} (μ={mean_err:.3f}, σ={std_err:.3f})")
        
        # ----------------------------------------------------
        # Panel 2: Mean Absolute Error per Horizon Step
        # ----------------------------------------------------
        # Mean absolute residual at each step t in {1 ... H}
        mae_per_step = np.mean(np.abs(res_chan), axis=0)
        time_steps = np.arange(1, pred_len + 1)
        
        ax2.plot(time_steps, mae_per_step, color=color, linewidth=2.0, label=f"{name}")

    # --- Formatting Panel 1 ---
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.7, label='Zero Bias Reference')
    ax1.set_title("1. Residual Density & Bias Distribution", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Residual Error (y - ŷ)")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)

    # --- Formatting Panel 2 ---
    ax2.set_title("2. Error Horizon Drift (MAE per Time Step)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Forecast Step Horizon (t)")
    ax2.set_ylabel("Mean Absolute Residual (|y - ŷ|)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)

    plt.suptitle(f"Multi-Model Residuals Comparison (Channel {channel_idx})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"--> Multi-model residual diagnostics saved to: {save_path}")

def summarize_and_plot_experiment(results_dict, 
                                 param_name="Horizon", 
                                 title="TimesNet Experiment Analysis",
                                 xlabel="Parameter", 
                                 save_path="./plots/experiment_results.png",
                                 plot_type="line", 
                                 prefix=""):
    """
    Unified function to print benchmark summary tables and plot results for:
      - Continuous sweeps (Horizons, top_k) -> Dual-axis MSE & MAE line plot
      - Categorical/Binary comparisons (RevIN) -> Side-by-side training trajectory plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Build Summary Table
    summary_data = []
    for key, res in results_dict.items():
        summary_data.append({
            param_name: f"{prefix}{key}",
            "Test MSE": f"{res['test_mse']:.4f}",
            "Test MAE": f"{res['test_mae']:.4f}",
            "Best Val MSE": f"{min(res['val_losses']):.4f}"
        })
        
    df_summary = pd.DataFrame(summary_data)
    print("\n==================================================")
    print(f"       {title.upper()} SUMMARY")
    print("==================================================")
    print(df_summary.to_string(index=False))
    
    # Special calculation for binary comparisons like RevIN
    if "With_RevIN" in results_dict and "Without_RevIN" in results_dict:
        mse_with = results_dict["With_RevIN"]["test_mse"]
        mse_without = results_dict["Without_RevIN"]["test_mse"]
        pct = ((mse_without - mse_with) / mse_without) * 100
        print(f"\n--> RevIN Reduced Test Error (MSE) by: {pct:.2f}%")
        
    print("==================================================\n")

    # 2. Plotting Logic
    if plot_type == "line":
        # Dual-axis line plot for numerical sweeps (Horizon, Top-k)
        keys_list = list(results_dict.keys())
        test_mses = [results_dict[k]['test_mse'] for k in keys_list]
        test_maes = [results_dict[k]['test_mae'] for k in keys_list]
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        color = 'tab:red'
        ax1.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax1.set_ylabel('Test MSE Loss', color=color, fontsize=11, fontweight='bold')
        line1 = ax1.plot(keys_list, test_mses, color=color, marker='o', linewidth=2.5, markersize=8, label='Test MSE')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Test MAE Loss', color=color, fontsize=11, fontweight='bold')
        line2 = ax2.plot(keys_list, test_maes, color=color, marker='s', linestyle='--', linewidth=2.0, markersize=8, label='Test MAE')
        ax2.tick_params(axis='y', labelcolor=color)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xticks(keys_list, [f"{prefix}{k}" for k in keys_list])
        plt.tight_layout()
        
    elif plot_type == "trajectories":
        # Side-by-side loss curves for categorical comparisons (RevIN On vs Off)
        fig, axes = plt.subplots(1, len(results_dict), figsize=(6 * len(results_dict), 5), sharey=True)
        if len(results_dict) == 1:
            axes = [axes]
            
        colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
        
        for idx, (setting_name, res) in enumerate(results_dict.items()):
            ax = axes[idx]
            c = colors[idx % len(colors)]
            epochs = range(1, len(res["train_losses"]) + 1)
            
            ax.plot(epochs, res["train_losses"], label="Train Loss", color=c, linestyle="-", marker="o", markersize=4)
            ax.plot(epochs, res["val_losses"], label="Val Loss", color=c, linestyle="--", marker="x", markersize=5)
            ax.set_title(setting_name.replace("_", " "), fontweight="bold")
            ax.set_xlabel("Epoch")
            if idx == 0:
                ax.set_ylabel("MSE Loss")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        plt.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"--> Saved plot to: {save_path}")


def plot_multi_model_horizon_comparison(all_horizon_results, 
                                        horizons=[96, 192, 336, 720], 
                                        save_path="./plots/multi_model_horizon_comparison.png"):
    """
    Handles structure: all_horizon_results[model_name][H] = res
    where H can be int (96) or str ('96'/'H96') and res is a result dict containing 'test_mse'.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # ----------------------------------------------------
    # 1. Safely Unpack [model_name][H] -> test_mse
    # ----------------------------------------------------
    normalized_data = {}
    
    for model_name, h_dict in all_horizon_results.items():
        if not isinstance(h_dict, dict):
            continue
            
        normalized_data[model_name] = {}
        
        for h_key, res in h_dict.items():
            # Clean key to integer (e.g. '96', 'H96', 'H=96' -> 96)
            clean_k = str(h_key).replace('H=', '').replace('H', '').strip()
            
            # ONLY process keys that are actual numeric horizons
            if not clean_k.isdigit():
                continue
                
            h_int = int(clean_k)
            
            # Extract Test MSE from result dictionary or float value
            if isinstance(res, dict) and 'test_mse' in res:
                normalized_data[model_name][h_int] = float(res['test_mse'])
            elif isinstance(res, (float, int, np.floating)):
                normalized_data[model_name][h_int] = float(res)

    # ----------------------------------------------------
    # 2. Build Summary Table
    # ----------------------------------------------------
    summary_rows = []
    for H in horizons:
        row = {"Horizon (H)": f"H={H}"}
        for model_name, h_data in normalized_data.items():
            if H in h_data:
                row[f"{model_name} (MSE)"] = f"{h_data[H]:.4f}"
            else:
                row[f"{model_name} (MSE)"] = "N/A"
        summary_rows.append(row)
        
    df_summary = pd.DataFrame(summary_rows)
    print("\n=========================================================================")
    print("       MULTI-MODEL HORIZON GENERALIZATION SUMMARY (TEST MSE)")
    print("=========================================================================")
    print(df_summary.to_string(index=False))
    print("=========================================================================\n")

    # ----------------------------------------------------
    # 3. Plot Comparison Line Chart
    # ----------------------------------------------------
    MODEL_COLORS = {
        "TimesNet": "tab:red",
        "DLinear": "tab:blue",
        "1D-ConvNet": "tab:orange",
        "LSTM": "tab:green"
    }

    plt.figure(figsize=(9, 5.5))
    
    for model_name, h_data in normalized_data.items():
        color = MODEL_COLORS.get(model_name, "tab:gray")
        
        valid_horizons = [H for H in horizons if H in h_data]
        valid_mses = [h_data[H] for H in valid_horizons]
        
        if not valid_mses:
            continue
            
        lw = 2.5 if model_name == "TimesNet" else 1.8
        ls = "-" if model_name == "TimesNet" else "--"
        marker = "o" if model_name == "TimesNet" else "s"
        
        plt.plot(valid_horizons, valid_mses, color=color, linestyle=ls, 
                 linewidth=lw, marker=marker, markersize=7, label=model_name)

    plt.title("Model Performance Degradation Across Horizon Expansion", fontsize=12, fontweight='bold')
    plt.xlabel("Forecast Horizon (H)", fontsize=11, fontweight='bold')
    plt.ylabel("Test MSE Loss", fontsize=11, fontweight='bold')
    plt.xticks(horizons, [f"H={H}" for H in horizons])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"--> Saved horizon comparison plot to: {save_path}")
       
def extract_and_plot_fft_periods(model, test_loader, device, top_k=5, 
                                 sampling_interval_hours=1, 
                                 save_path="./plots/fft_period_analysis.png"):
    """
    Extracts, converts, and visualizes the top-k Fourier periodic components 
    discovered by TimesNet on the Electricity dataset.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Grab a test batch
    batch_x, _ = next(iter(test_loader))
    batch_x = batch_x.to(device) # Shape: (B, L, C)
    
    B, L, C = batch_x.shape
    
    # Apply RevIN normalization if enabled on model to inspect clean signals
    if hasattr(model, 'revin') and getattr(model, 'use_revin', True):
        x_norm = model.revin(batch_x, 'norm')
    else:
        x_norm = batch_x
        
    # 2. Compute Real FFT across time dimension L
    # x_ft shape: (B, L//2 + 1, C)
    x_ft = torch.fft.rfft(x_norm, dim=1)
    
    # Average frequency amplitudes across batch (B) and channel (C) dimensions
    amplitudes = torch.abs(x_ft).mean(dim=(0, 2)).cpu().numpy() # Shape: (L//2 + 1,)
    
    # Frequency indices (0, 1, ..., L//2)
    freq_indices = np.arange(len(amplitudes))
    
    # Zero out DC component (frequency 0 = baseline mean offset)
    amplitudes[0] = 0
    
    # 3. Find top-k dominant frequency indices
    top_k_indices = np.argsort(amplitudes)[-top_k:][::-1]
    top_k_amplitudes = amplitudes[top_k_indices]
    
    # Calculate period lengths: Period = Total Sequence Length / Frequency Index
    # p_i = L / f_i
    period_lengths_steps = [L / idx if idx > 0 else L for idx in top_k_indices]
    period_lengths_hours = [p * sampling_interval_hours for p in period_lengths_steps]
    
    # ----------------------------------------------------
    # 4. Print Summary Table of Discovered Periods
    # ----------------------------------------------------
    print("\n==================================================")
    print(f"   TIMESNET FFT DOMAIN INTERPRETABILITY (L={L})")
    print("==================================================")
    print(f"{'Rank':<6} | {'Freq Index':<12} | {'Period (Steps)':<16} | {'Period (Hours)':<16} | {'Amplitude':<10}")
    print("-" * 72)
    
    for rank, (idx, p_step, p_hr, amp) in enumerate(zip(top_k_indices, period_lengths_steps, period_lengths_hours, top_k_amplitudes), 1):
        print(f"{rank:<6} | {idx:<12} | {p_step:<16.2f} | {p_hr:<16.2f}h | {amp:<10.4f}")
    print("==================================================\n")

    # ----------------------------------------------------
    # 5. Visualization: FFT Amplitude Spectrum & Periods
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5))
    
    # Full FFT Spectrum Line
    plt.plot(freq_indices[1:], amplitudes[1:], color='tab:blue', linewidth=1.8, label='FFT Amplitude Spectrum')
    
    # Highlight Top-k Frequency Peaks
    plt.scatter(top_k_indices, top_k_amplitudes, color='tab:red', s=70, zorder=5, label=f'Top-{top_k} Selected Periods')
    
    # Annotate Top Peaks with physical period length
    for idx, p_hr, amp in zip(top_k_indices, period_lengths_hours, top_k_amplitudes):
        plt.annotate(
            f"T={p_hr:.1f}h\n(f={idx})", 
            xy=(idx, amp), 
            xytext=(idx + 0.5, amp * 1.05),
            fontsize=9, fontweight='bold', color='tab:red',
            arrowprops=dict(arrowstyle="->", color='tab:red', lw=1.0)
        )
        
    plt.title(f"TimesNet FFT Frequency Spectrum & Discovered Periods (Electricity, L={L})", fontsize=12, fontweight='bold')
    plt.xlabel("Frequency Index (f)", fontsize=11)
    plt.ylabel("Mean Spectral Amplitude", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"--> Saved FFT Period Analysis Plot to: {save_path}")

def plot_parametric_gaussian_forecast(model, test_loader, device, sample_idx=0, channel_idx=0, 
                                     seq_len=96, pred_len=96, save_path="./plots/gaussian_forecast.png"):
    model.eval()
    batch_x, batch_y = next(iter(test_loader))
    batch_x = batch_x.to(device)
    
    lookback = batch_x[sample_idx, :, channel_idx].cpu().numpy()
    ground_truth = batch_y[sample_idx, -pred_len:, channel_idx].cpu().numpy()
    
    with torch.no_grad():
        mu, sigma = model(batch_x)
        mu_pred = mu[sample_idx, :, channel_idx].cpu().numpy()
        sigma_pred = sigma[sample_idx, :, channel_idx].cpu().numpy()

    # Calculate 1.645 * sigma for 90% Confidence Interval (Z_0.95 = 1.645)
    # Calculate 2.576 * sigma for 99% Confidence Interval
    upper_90 = mu_pred + 1.645 * sigma_pred
    lower_90 = mu_pred - 1.645 * sigma_pred
    
    upper_99 = mu_pred + 2.576 * sigma_pred
    lower_99 = mu_pred - 2.576 * sigma_pred

    time_lookback = np.arange(seq_len)
    time_horizon = np.arange(seq_len, seq_len + pred_len)

    plt.figure(figsize=(12, 5))
    
    # Ground Truth & History
    plt.plot(time_lookback, lookback, color='black', label='Historical Lookback')
    plt.plot(time_horizon, ground_truth, color='black', linestyle=':', label='Ground Truth', linewidth=2)
    
    # Predicted Mean
    plt.plot(time_horizon, mu_pred, color='tab:red', label='Predicted Mean (μ)', linewidth=2)
    
    # 90% Confidence Band
    plt.fill_between(time_horizon, lower_90, upper_90, color='tab:red', alpha=0.30, label='90% Confidence Region')
    
    # 99% Confidence Band (Outer edge)
    plt.fill_between(time_horizon, lower_99, upper_99, color='tab:red', alpha=0.10, label='99% Confidence Region')

    plt.axvline(x=seq_len, color='gray', linestyle='--', alpha=0.7)
    plt.title(f"TimesNet Parametric Gaussian Probabilistic Forecast (Channel {channel_idx})", fontweight='bold')
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Value")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def predict_mc_dropout(model, x, num_samples=50, dropout_rate=0.3):
    """
    Performs Monte Carlo Dropout sampling at test time.
    
    Args:
        model: Trained PyTorch TimesNet model.
        x: Input tensor [B, seq_len, enc_in].
        num_samples: Number of forward stochastic passes (default 50).
        
    Returns:
        mean_pred: Mean predicted values across samples [B, pred_len, enc_in].
        lower_bound: 5th percentile boundary (90% CI lower) [B, pred_len, enc_in].
        upper_bound: 95th percentile boundary (90% CI upper) [B, pred_len, enc_in].
        all_samples: Raw sampled predictions tensor [num_samples, B, pred_len, enc_in].
    """
    model.eval()
    
    # Enable dropout modules specifically during inference
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            module.p = dropout_rate
            
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(x) # Shape: [B, pred_len, enc_in]
            samples.append(pred.cpu().numpy())
            
    # Shape: [num_samples, B, pred_len, enc_in]
    all_samples = np.array(samples)
    
    # Calculate statistics across the sample dimension (axis 0)
    mean_pred = np.mean(all_samples, axis=0)
    lower_bound = np.percentile(all_samples, 5, axis=0)  # 90% CI lower
    upper_bound = np.percentile(all_samples, 95, axis=0)  # 90% CI upper
    
    return mean_pred, lower_bound, upper_bound, all_samples

def plot_mc_dropout_forecast(model, test_loader, device, sample_idx=0, channel_idx=0, 
                             num_samples=100, num_top_paths=5, save_path="./plots/mc_dropout_forecast.png"):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Fetch test sample batch
    batch_x, batch_y = next(iter(test_loader))
    batch_x = batch_x.to(device)
    
    seq_len = model.seq_len if hasattr(model, 'seq_len') else batch_x.shape[1]
    pred_len = model.pred_len if hasattr(model, 'pred_len') else batch_y.shape[1]
    
    # Run MC Dropout sampling
    mean_pred, lower, upper, all_samples = predict_mc_dropout(model, batch_x, num_samples=num_samples)
    
    # Extract specific channel and sample index
    lookback = batch_x[sample_idx, :, channel_idx].cpu().numpy()
    ground_truth = batch_y[sample_idx, -pred_len:, channel_idx].cpu().numpy()
    
    mean_curve = mean_pred[sample_idx, :, channel_idx]
    lower_curve = lower[sample_idx, :, channel_idx]
    upper_curve = upper[sample_idx, :, channel_idx]
    
    # Extract top candidate trajectory samples: shape [num_top_paths, pred_len]
    top_trajectories = all_samples[:num_top_paths, sample_idx, :, channel_idx]

    time_lookback = np.arange(seq_len)
    time_horizon = np.arange(seq_len, seq_len + pred_len)

    plt.figure(figsize=(12, 5))
    
    # 1. Historical Lookback & Ground Truth
    plt.plot(time_lookback, lookback, color='black', label='Historical Lookback', linewidth=1.8)
    plt.plot(time_horizon, ground_truth, color='black', linestyle=':', label='Ground Truth', linewidth=2.2)
    
    # 2. Plot Top-5 Candidate Sample Paths
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, num_top_paths))
    for i in range(num_top_paths):
        label = "Sampled Trajectories" if i == 0 else None
        plt.plot(time_horizon, top_trajectories[i], color=colors[i], linestyle='--', alpha=0.75, linewidth=1.2, label=label)
        
    # 3. Plot Mean/Median Trajectory
    plt.plot(time_horizon, mean_curve, color='tab:red', label='MC Mean Prediction (μ)', linewidth=2.5)
    
    # 4. Plot Epistemic Uncertainty Ribbon
    plt.fill_between(time_horizon, lower_curve, upper_curve, color='tab:red', alpha=0.20, label='90% MC Confidence Band')

    plt.axvline(x=seq_len, color='gray', linestyle='--', alpha=0.7)
    plt.title(f"TimesNet Monte Carlo Dropout Probabilistic Forecast (Channel {channel_idx})", fontweight='bold')
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Value")
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"--> Saved MC Dropout forecast plot to: {save_path}")