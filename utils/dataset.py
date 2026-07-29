import os
import urllib.request
import gzip
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def download_datasets(data_dir: str = "./data"):
    """
    Downloads electricity.txt.gz and exchange_rate.txt.gz directly from 
    laiguokun/multivariate-time-series-data GitHub repository.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    datasets = {
        # Electricity (ECL): 321 channels, hourly load
        "electricity.txt.gz": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz",
        
        # Financial Benchmark: Exchange Rate (8 foreign currencies, daily steps)
        "exchange_rate.txt.gz": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz"
    }
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    
    for filename, url in datasets.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"--> Successfully downloaded {filename}")
            except Exception as e:
                print(f"--> Error downloading {filename}: {e}")
        else:
            print(f"--> Found existing {filename} in {data_dir}")

class Dataset_Custom(Dataset):
    def __init__(self, root_path="./data", data_path='electricity.txt.gz', flag='train', 
                 size=None, scale=True):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)
        
        # Decompress and read .txt.gz numerical matrix
        with gzip.open(file_path, 'rt') as f:
            df_data = pd.read_csv(f, sep=r'\s+|,', header=None, engine='python')

        # 70% Train, 10% Val, 20% Test
        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_vali = len(df_data) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_data)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Fit feature scaling exclusively on training split (no leakage)
        if self.scale:
            train_data = df_data.values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    
def print_dataset_summary(data_path="./data/electricity.txt.gz"):
    """
    Computes and prints basic structural and statistical metrics for a time series dataset.
    """
    filename = os.path.basename(data_path)
    
    # Read matrix file
    if data_path.endswith('.gz'):
        with gzip.open(data_path, 'rt') as f:
            df = pd.read_csv(f, sep=r'\s+|,', header=None, engine='python')
    else:
        df = pd.read_csv(data_path)

    data = df.values
    
    # Metric Calculations
    num_timesteps, num_channels = data.shape
    total_points = data.size
    missing_values = df.isnull().sum().sum()
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    
    print(f"=" * 45)
    print(f"       DATASET METRICS SUMMARY: {filename}")
    print(f"=" * 45)
    print(f"• Total Time Steps (T):      {num_timesteps:,}")
    print(f"• Number of Channels/Variates (C): {num_channels}")
    print(f"• Total Data Points (T x C):  {total_points:,}")
    print(f"• Missing Values:            {missing_values}")
    print(f"• Global Mean:              {mean_val:.4f}")
    print(f"• Global Std Dev:           {std_val:.4f}")
    print(f"• Global Min / Max Range:   [{min_val:.4f}, {max_val:.4f}]")
    print(f"=" * 45 + "\n")
    
def data_provider(args, flag):
    """
    Factory function to instantiate Dataset_Custom and DataLoader.
    """
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = getattr(args, 'batch_size', 32)
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = getattr(args, 'batch_size', 32)

    dataset = Dataset_Custom(
        root_path=getattr(args, 'root_path', './data'),
        data_path=getattr(args, 'data_path', 'electricity.txt.gz'),
        flag=flag,
        size=[getattr(args, 'seq_len', 96), getattr(args, 'label_len', 48), getattr(args, 'pred_len', 96)],
        scale=True
    )
    
    # num_workers=0 ensures seamless execution on macOS (MPS)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last
    )
    return dataset, data_loader