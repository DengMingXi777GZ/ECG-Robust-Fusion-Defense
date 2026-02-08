"""
MIT-BIH 心律失常数据集加载器
支持：Kaggle MIT-BIH 数据集 (187采样点 + 标签)
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class MITBIHDataset(Dataset):
    """
    MIT-BIH 心律失常数据集
    
    Args:
        csv_path: CSV文件路径 (包含187列ECG信号 + 1列标签)
        transform: 可选变换 ('normalize' 或其他)
        preload: 是否预加载全部数据到内存
    """
    
    def __init__(self, csv_path, transform='normalize', preload=True):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.preload = preload
        
        # 标签映射: MIT-BIH 5类 (N,S,V,F,Q) -> 0-4
        # 数据集中的标签已经是0-4整数
        self.label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        
        if preload:
            self._load_data()
        else:
            # 仅获取长度信息，延迟加载
            df = pd.read_csv(csv_path, nrows=1)
            self.data_len = len(pd.read_csv(csv_path))
    
    def _load_data(self):
        """加载并预处理数据"""
        print(f"Loading data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path, header=None)
        
        # 最后一列为标签
        self.labels = df.iloc[:, -1].values.astype(np.int64)
        self.signals = df.iloc[:, :-1].values.astype(np.float32)
        
        # 信号形状: [N, 187] -> [N, 1, 187] (通道优先)
        self.signals = self.signals.reshape(-1, 1, 187)
        
        # 归一化: Min-Max 到 [0, 1]
        if self.transform == 'normalize':
            self.signals = self._normalize(self.signals)
        
        self.data_len = len(self.labels)
        print(f"Loaded {self.data_len} samples")
    
    def _normalize(self, signals):
        """Min-Max 归一化到 [0, 1]"""
        # 对每个样本单独归一化
        signals_min = signals.min(axis=2, keepdims=True)
        signals_max = signals.max(axis=2, keepdims=True)
        
        # 避免除零
        denom = signals_max - signals_min
        denom[denom == 0] = 1.0
        
        return (signals - signals_min) / denom
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        if self.preload:
            x = torch.from_numpy(self.signals[idx].copy())
            y = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            # 延迟加载模式
            row = pd.read_csv(self.csv_path, skiprows=idx, nrows=1, header=None)
            y = torch.tensor(int(row.iloc[0, -1]), dtype=torch.long)
            x = torch.tensor(row.iloc[0, :-1].values.astype(np.float32))
            x = x.unsqueeze(0)  # [1, 187]
            
            if self.transform == 'normalize':
                x_min, x_max = x.min(), x.max()
                if x_max > x_min:
                    x = (x - x_min) / (x_max - x_min)
        
        return x, y
    
    def get_class_distribution(self):
        """返回类别分布统计"""
        if self.preload:
            unique, counts = np.unique(self.labels, return_counts=True)
            return dict(zip(unique, counts))
        else:
            return "Data not preloaded, distribution unavailable"


def get_mitbih_loaders(train_csv='data/mitbih_train.csv', 
                       test_csv='data/mitbih_test.csv',
                       batch_size=32, 
                       num_workers=0):
    """
    便捷函数：获取训练和测试DataLoader
    
    Returns:
        train_loader, test_loader
    """
    train_set = MITBIHDataset(train_csv, transform='normalize', preload=True)
    test_set = MITBIHDataset(test_csv, transform='normalize', preload=True)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # 简单测试
    print("Testing MITBIHDataset...")
    
    # 创建模拟数据用于测试
    import tempfile
    import os
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 100
    mock_data = np.random.randn(n_samples, 187).astype(np.float32)
    mock_labels = np.random.randint(0, 5, n_samples)
    mock_full = np.column_stack([mock_data, mock_labels])
    
    # 保存临时文件
    temp_path = tempfile.mktemp(suffix='.csv')
    np.savetxt(temp_path, mock_full, delimiter=',')
    
    try:
        # 测试数据集
        dataset = MITBIHDataset(temp_path, transform='normalize', preload=True)
        print(f"Dataset size: {len(dataset)}")
        print(f"Class distribution: {dataset.get_class_distribution()}")
        
        # 测试DataLoader
        loader = DataLoader(dataset, batch_size=32)
        x, y = next(iter(loader))
        print(f"Batch x shape: {x.shape}")  # [32, 1, 187]
        print(f"Batch y shape: {y.shape}")  # [32]
        print(f"X range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"Y unique: {torch.unique(y)}")
        
        print("\n[OK] All tests passed!")
        
    finally:
        os.remove(temp_path)
