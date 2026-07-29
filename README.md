# General Time Series Forecasting: TimesNet vs. Baselines

This repository contains the implementation, baseline comparisons, and experiments for the final project based on the paper **"TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"** (ICLR 2023).

---

## Project Overview

Time series forecasting requires capturing complex temporal dynamics. While standard models operate purely in 1D space, TimesNet transforms 1D time series into 2D space based on dominant periodicities discovered via Fast Fourier Transform (FFT). By applying 2D convolutions, it concurrently captures both intraperiod-variations (short-term local patterns) and interperiod-variations (long-term trend dynamics across consecutive periods).

We evaluate TimesNet on the Forecasting task across two distinct dataset profiles:
1. **Strongly Periodic Data:** Electricity Consumption.
2. **Weakly Periodic / Trend-Dominated Data:** Financial Dataset.

We compare TimesNet against lightweight and standard 1D baselines (`DLinear` and `1D-ConvNet`) across multiple forecasting horizons ($H \in \{96, 192, 336\}$).

---

## Hardware & Acceleration Support

This project dynamically configures PyTorch to use available hardware acceleration:
* **Apple Silicon (M5 / MPS):** Uses Metal Performance Shaders (`mps`).
* **NVIDIA GPUs:** Uses CUDA (`cuda`).
* **Fallback:** Standard CPU execution (`cpu`).

---

## Project Structure

```text
time_series_project/
├── main.ipynb             # Primary notebook executing data load, training, and plots
├── data/                  
│   ├── Electricity.csv
│   └── Exchange_Rate.csv
├── models/
│   ├── __init__.py
│   ├── dlinear.py         # Baseline 1: Channel-wise Linear Mapping
│   ├── conv1d_baseline.py # Baseline 2: 1D Convolutional Network
│   └── timesnet.py        # TimesNet Architecture 
├── utils/
│   ├── dataset.py         # PyTorch Dataset and DataLoader windowing
│   └── visualization.py   # Plotting functions for report figures
├── train.py               # Training and validation loop script
├── requirements.txt       # Dependencies list
└── README.md              # Project documentation