import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.dataset import data_provider

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Executes one training epoch with gradient clipping.
    """
    model.train()
    total_loss = []
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        
        batch_x = batch_x.to(device)
        pred_len = model.pred_len if hasattr(model, 'pred_len') else batch_x.shape[1]
        target_y = batch_y[:, -pred_len:, :].to(device)
        
        output = model(batch_x)
        loss = criterion(output, target_y)
        loss.backward()
        
        # Gradient clipping for stabilization
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        total_loss.append(loss.item())
        
    return np.mean(total_loss)


def evaluate(model, data_loader, criterion, metric_mae, device):
    """
    Evaluates model performance on Validation or Test dataset split.
    """
    model.eval()
    total_mse = []
    total_mae = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            pred_len = model.pred_len if hasattr(model, 'pred_len') else batch_x.shape[1]
            target_y = batch_y[:, -pred_len:, :].to(device)
            
            output = model(batch_x)
            
            mse = criterion(output, target_y).item()
            mae = metric_mae(output, target_y).item()
            
            total_mse.append(mse)
            total_mae.append(mae)
            
    return np.mean(total_mse), np.mean(total_mae)


def train_model(model, args, device):
    """
    Enhanced Training Loop with Learning Rate Scheduling and Early Stopping.
    """
    train_dataset, train_loader = data_provider(args, flag='train')
    val_dataset, val_loader = data_provider(args, flag='val')
    test_dataset, test_loader = data_provider(args, flag='test')
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=getattr(args, 'learning_rate', 1e-3), 
        weight_decay=1e-4
    )
    
    # Decays LR by 0.5 if validation loss doesn't improve for 2 consecutive epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    criterion = nn.MSELoss()
    metric_mae = nn.L1Loss()
    
    best_val_loss = float('inf')
    best_model_weights = None
    
    train_losses = []
    val_losses = []
    val_maes = []
    
    print(f"\n==================================================")
    print(f"       START TRAINING: {args.model_name}")
    print(f"       Dataset: {args.data_path} | Device: {device}")
    print(f"==================================================")
    
    start_time = time.time()
    
    for epoch in range(1, args.train_epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        val_mse, val_mae = evaluate(model, val_loader, criterion, metric_mae, device)
        
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        val_losses.append(val_mse)
        val_maes.append(val_mae)
        
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{args.train_epochs:02d} | Train MSE: {train_loss:.4f} | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f} | LR: {current_lr:.1e} | Time: {elapsed:.2f}s")
        
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_model_weights = copy.deepcopy(model.state_dict())
                
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s.")
    
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    test_mse, test_mae = evaluate(model, test_loader, criterion, metric_mae, device)
    
    print(f"--------------------------------------------------")
    print(f"--> TEST RESULTS ({args.model_name}):")
    print(f"    • Test MSE: {test_mse:.4f}")
    print(f"    • Test MAE: {test_mae:.4f}")
    print(f"--------------------------------------------------\n")
    
    results = {
        "model": args.model_name,
        "dataset": args.data_path,
        "pred_len": args.pred_len,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_maes": val_maes
    }
    
    return results

def train_epoch_probabilistic(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = []
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        
        batch_x = batch_x.to(device)
        target_y = batch_y[:, -model.pred_len:, :].to(device)
        
        # Forward pass yields predicted Mean (mu) and Std (sigma)
        mu, sigma = model(batch_x)
        
        # Gaussian NLL Loss expects var = sigma**2
        # Composite Loss: MSE ensures sharpness of mu, NLL trains uncertainty sigma
        mse_loss = F.mse_loss(mu, target_y)
        nll_loss = criterion(mu, target_y, torch.square(sigma))

        loss = nll_loss + 2.0 * mse_loss  # Weighting factor keeps mu tracking ground truth spikes
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss.append(loss.item())
        
    return np.mean(total_loss)


def evaluate_probabilistic(model, val_loader, criterion, device):
    model.eval()
    total_loss = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            target_y = batch_y[:, -model.pred_len:, :].to(device)
            
            mu, sigma = model(batch_x)
            loss = criterion(mu, target_y, torch.square(sigma))
            total_loss.append(loss.item())
            
    return np.mean(total_loss)

def train_probabilistic_model(model, args, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.GaussianNLLLoss()
    
    
    best_val_loss = float('inf')
    best_weights = None
    
    print(f"--> Training Probabilistic TimesNet on {device} ({args.train_epochs} epochs)...")
    
    for epoch in range(1, args.train_epochs + 1):
        train_nll = train_epoch_probabilistic(model, train_loader, optimizer, criterion, device)
        val_nll = evaluate_probabilistic(model, val_loader, criterion, device)
        
        if val_nll < best_val_loss:
            best_val_loss = val_nll
            best_weights = copy.deepcopy(model.state_dict())
            
        print(f"Epoch {epoch:02d}/{args.train_epochs:02d} | Train loss: {train_nll:.4f} | Val loss: {val_nll:.4f}")
        if device.type == 'mps':
            torch.mps.empty_cache()
        
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return {'val_nll': best_val_loss}


