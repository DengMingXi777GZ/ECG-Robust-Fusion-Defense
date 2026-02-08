"""
融合数据集 (Fusion Dataset)
同时提供 ECG 信号和手工特征
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class FusionDataset(Dataset):
    """
    双分支融合网络的数据集
    同时返回: (signal, handcrafted_features, label)
    """
    
    def __init__(self, signals, labels, handcrafted_features=None, 
                 handcrafted_path=None, normalize=True):
        """
        Args:
            signals: [N, 1, 187] 或 [N, 187] ECG 信号
            labels: [N] 标签
            handcrafted_features: [N, 12] 预计算的手工特征
            handcrafted_path: 预提取特征的 .npy 文件路径
            normalize: 是否对手工特征进行 Z-score 标准化
        """
        # 确保信号形状为 [N, 1, 187]
        if len(signals.shape) == 2:
            self.signals = signals.reshape(-1, 1, 187).astype(np.float32)
        else:
            self.signals = signals.astype(np.float32)
        
        self.labels = labels.astype(np.int64)
        
        # 加载或设置手工特征
        if handcrafted_features is not None:
            self.handcrafted = handcrafted_features.astype(np.float32)
        elif handcrafted_path and os.path.exists(handcrafted_path):
            self.handcrafted = np.load(handcrafted_path).astype(np.float32)
        else:
            raise ValueError("必须提供 handcrafted_features 或有效的 handcrafted_path")
        
        # 验证维度
        assert len(self.signals) == len(self.labels) == len(self.handcrafted), \
            f"维度不匹配: signals={len(self.signals)}, labels={len(self.labels)}, " \
            f"handcrafted={len(self.handcrafted)}"
        
        assert self.handcrafted.shape[1] == 12, \
            f"手工特征维度错误: {self.handcrafted.shape[1]} != 12"
        
        # 标准化
        if normalize:
            self.handcrafted = self._normalize_handcrafted(self.handcrafted)
        
        print(f"[FusionDataset] 加载 {len(self.signals)} 条样本")
        print(f"  - 信号: {self.signals.shape}")
        print(f"  - 手工特征: {self.handcrafted.shape}")
        print(f"  - 标签: {self.labels.shape}")
    
    def _normalize_handcrafted(self, features):
        """对手工特征进行 Z-score 标准化"""
        # 使用训练集的统计量（这里简化为对整个数据集）
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std == 0, 1.0, std)  # 避免除以零
        
        normalized = (features - mean) / std
        return normalized
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.signals[idx]),           # [1, 187]
            torch.from_numpy(self.handcrafted[idx]),       # [12]
            torch.tensor(self.labels[idx], dtype=torch.long)  # scalar
        )


class FusionDatasetWithAdversarial(Dataset):
    """
    包含对抗样本的融合数据集
    用于检测器的训练
    """
    
    def __init__(self, clean_signals, clean_features, clean_labels,
                 adv_signals, adv_features, adv_labels=None):
        """
        Args:
            clean_signals: [N, 1, 187] 干净信号
            clean_features: [N, 12] 干净特征
            clean_labels: [N] 干净标签
            adv_signals: [M, 1, 187] 对抗信号
            adv_features: [M, 12] 对抗特征
            adv_labels: [M] 对抗标签 (可选，默认为 -1 表示对抗)
        """
        # 合并数据
        self.signals = np.vstack([clean_signals, adv_signals]).astype(np.float32)
        self.handcrafted = np.vstack([clean_features, adv_features]).astype(np.float32)
        
        # 标签：干净样本 = 类别标签, 对抗样本 = -1
        if adv_labels is None:
            adv_labels = np.full(len(adv_signals), -1, dtype=np.int64)
        
        self.labels = np.concatenate([clean_labels, adv_labels]).astype(np.int64)
        
        # 是否是对抗样本的标记 (用于检测器训练)
        self.is_adversarial = np.concatenate([
            np.zeros(len(clean_signals), dtype=np.float32),
            np.ones(len(adv_signals), dtype=np.float32)
        ])
        
        # 标准化
        self.handcrafted = self._normalize_handcrafted(self.handcrafted)
        
        print(f"[FusionDatasetWithAdversarial] 加载 {len(self.signals)} 条样本")
        print(f"  - 干净样本: {len(clean_signals)}")
        print(f"  - 对抗样本: {len(adv_signals)}")
    
    def _normalize_handcrafted(self, features):
        """对手工特征进行 Z-score 标准化"""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std == 0, 1.0, std)
        return (features - mean) / std
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.signals[idx]),
            torch.from_numpy(self.handcrafted[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.is_adversarial[idx], dtype=torch.float32)
        )


def load_mitbih_data_for_fusion(data_dir='data', use_precomputed=True):
    """
    加载 MIT-BIH 数据集用于融合模型
    
    Returns:
        train_dataset: FusionDataset
        test_dataset: FusionDataset
    """
    print("=" * 60)
    print("加载 MIT-BIH 数据集 (用于融合模型)")
    print("=" * 60)
    
    # 加载训练和测试数据
    train_df = pd.read_csv(os.path.join(data_dir, 'mitbih_train.csv'), header=None)
    test_df = pd.read_csv(os.path.join(data_dir, 'mitbih_test.csv'), header=None)
    
    # 分离特征和标签
    X_train = train_df.iloc[:, :-1].values.reshape(-1, 1, 187).astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.int64)
    
    X_test = test_df.iloc[:, :-1].values.reshape(-1, 1, 187).astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int64)
    
    # 加载预计算的手工特征
    if use_precomputed and os.path.exists(os.path.join(data_dir, 'handcrafted_features_test.npy')):
        print("\n加载预计算的手工特征...")
        train_features = None
        test_features = np.load(os.path.join(data_dir, 'handcrafted_features_test.npy'))
        
        # 如果需要训练集特征，可以在这里提取
        # 暂时使用测试特征作为占位（实际应该提取训练集特征）
        if not os.path.exists(os.path.join(data_dir, 'handcrafted_features_train.npy')):
            print("[Warning] 训练集手工特征不存在，需要提取...")
            from features.ecg_features import ECGFeatureExtractor
            extractor = ECGFeatureExtractor()
            train_features = extractor.extract_batch(X_train.squeeze(1), verbose=True)
            np.save(os.path.join(data_dir, 'handcrafted_features_train.npy'), train_features)
        else:
            train_features = np.load(os.path.join(data_dir, 'handcrafted_features_train.npy'))
    else:
        print("\n提取手工特征（这可能需要几分钟）...")
        from features.ecg_features import ECGFeatureExtractor
        extractor = ECGFeatureExtractor()
        
        print("提取训练集特征...")
        train_features = extractor.extract_batch(X_train.squeeze(1), verbose=True)
        
        print("提取测试集特征...")
        test_features = extractor.extract_batch(X_test.squeeze(1), verbose=True)
        
        # 保存
        np.save(os.path.join(data_dir, 'handcrafted_features_train.npy'), train_features)
        np.save(os.path.join(data_dir, 'handcrafted_features_test.npy'), test_features)
    
    # 创建数据集
    train_dataset = FusionDataset(
        signals=X_train,
        labels=y_train,
        handcrafted_features=train_features,
        normalize=True
    )
    
    test_dataset = FusionDataset(
        signals=X_test,
        labels=y_test,
        handcrafted_features=test_features,
        normalize=True
    )
    
    return train_dataset, test_dataset


def load_adversarial_for_fusion(attack='pgd', eps=0.05, data_dir='data'):
    """
    加载对抗样本用于融合模型测试
    
    Returns:
        adv_signals: [N, 1, 187]
        adv_features: [N, 12]
        labels: [N]
    """
    # 加载对抗样本
    adv_path = os.path.join(data_dir, f'adversarial/eps{eps:.0e}/test_{attack}.pt')
    
    if not os.path.exists(adv_path):
        # 尝试旧路径
        adv_path = os.path.join(data_dir, f'adversarial/test_{attack}.pt')
    
    adv_data = torch.load(adv_path, map_location='cpu', weights_only=False)
    adv_signals = adv_data['x_adv'].numpy()
    labels = adv_data['y'].numpy().astype(np.int64)
    
    # 提取或加载手工特征
    features_path = os.path.join(data_dir, f'handcrafted_features_{attack}.npy')
    
    if os.path.exists(features_path):
        print(f"加载 {attack} 对抗样本预计算特征...")
        adv_features = np.load(features_path)
    else:
        print(f"提取 {attack} 对抗样本特征...")
        from features.ecg_features import ECGFeatureExtractor
        extractor = ECGFeatureExtractor()
        adv_features = extractor.extract_batch(adv_signals.squeeze(1), verbose=True)
        np.save(features_path, adv_features)
    
    return adv_signals, adv_features, labels


def test_fusion_dataset():
    """测试融合数据集"""
    print("=" * 60)
    print("融合数据集测试")
    print("=" * 60)
    
    # 创建模拟数据
    n_samples = 100
    signals = np.random.randn(n_samples, 1, 187).astype(np.float32)
    labels = np.random.randint(0, 5, n_samples).astype(np.int64)
    handcrafted = np.random.randn(n_samples, 12).astype(np.float32)
    
    # 创建数据集
    dataset = FusionDataset(
        signals=signals,
        labels=labels,
        handcrafted_features=handcrafted,
        normalize=True
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 获取一个样本
    signal, hc_feat, label = dataset[0]
    
    print(f"\n样本形状:")
    print(f"  信号: {signal.shape} (类型: {signal.dtype})")
    print(f"  手工特征: {hc_feat.shape} (类型: {hc_feat.dtype})")
    print(f"  标签: {label} (类型: {type(label)})")
    
    # 验证标准化
    print(f"\n手工特征统计:")
    print(f"  均值: {torch.mean(hc_feat).item():.4f}")
    print(f"  标准差: {torch.std(hc_feat).item():.4f}")
    
    # 测试 DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for batch_signals, batch_hc, batch_labels in loader:
        print(f"\n批次形状:")
        print(f"  信号: {batch_signals.shape}")
        print(f"  手工特征: {batch_hc.shape}")
        print(f"  标签: {batch_labels.shape}")
        break
    
    print("\n" + "=" * 60)
    print("[OK] 融合数据集测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_fusion_dataset()
