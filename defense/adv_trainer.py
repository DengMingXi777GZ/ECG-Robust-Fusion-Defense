"""
对抗训练数据集生成器 (Adversarial Training Dataset)
Task 2.1: 动态生成对抗样本，节省内存
"""
import torch
from torch.utils.data import Dataset
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks import PGD, SAP


class AdversarialDataset(Dataset):
    """
    动态生成对抗样本的数据集
    
    关键特性:
        - 每次getitem生成新的对抗样本（避免过拟合到固定噪声）
        - 直接复用Layer 1的PGD/SAP实现
        - 支持eps=0.05（基于Layer 1验证的有效参数）
    
    Args:
        clean_dataset: 原始干净数据集
        model: 用于生成对抗样本的模型
        attack_method: 攻击方法 ('pgd' 或 'sap')
        eps: 扰动预算 (默认0.05)
        steps: 攻击步数 (默认40)
        alpha: 步长 (默认 eps/4 = 0.0125)
        device: 计算设备
    """
    
    def __init__(self, clean_dataset, model, attack_method='pgd',
                 eps=0.05, steps=40, alpha=None, device='cuda'):
        """
        初始化对抗数据集
        
        关键：使用Layer 1验证有效的eps=0.05参数
        """
        self.clean_data = clean_dataset
        self.model = model
        self.device = device
        self.eps = eps
        
        # 默认alpha = eps/4
        if alpha is None:
            alpha = eps / 4
        
        # 直接复用Layer 1的attacks模块
        if attack_method == 'pgd':
            self.attacker = PGD(model, device=device, eps=eps, num_steps=steps, alpha=alpha, random_start=True)
            self.attack_name = 'PGD'
        elif attack_method == 'sap':
            self.attacker = SAP(model, device=device, eps=eps, num_steps=steps, use_pgd_init=True)
            self.attack_name = 'SAP'
        else:
            raise ValueError(f"Unknown attack method: {attack_method}. Use 'pgd' or 'sap'.")
        
        print(f"[AdversarialDataset] Initialized with {self.attack_name} attack")
        print(f"  - eps={eps}, steps={steps}, alpha={alpha:.4f}")
        print(f"  - Dataset size: {len(clean_dataset)}")
    
    def __len__(self):
        return len(self.clean_data)
    
    def __getitem__(self, idx):
        """
        获取数据样本
        
        Returns:
            x_clean: 干净样本 [1, 187]
            x_adv: 对抗样本 [1, 187]  
            y: 标签 (scalar)
        """
        x, y = self.clean_data[idx]
        
        # 确保数据在正确的设备上
        x = x.to(self.device)
        y = torch.tensor(y, device=self.device) if not isinstance(y, torch.Tensor) else y.to(self.device)
        
        # 关键：每次getitem生成新的对抗样本（避免过拟合到固定噪声）
        # 添加batch维度 -> [1, 1, 187]
        x_batch = x.unsqueeze(0)
        y_batch = y.unsqueeze(0) if y.dim() == 0 else y.view(1)
        
        with torch.enable_grad():  # 确保梯度开启
            x_adv = self.attacker.generate(x_batch, y_batch).squeeze(0)
        
        # 返回CPU张量以兼容DataLoader的多进程
        return x.cpu(), x_adv.cpu(), y.cpu()


class MixedAdversarialDataset(Dataset):
    """
    混合数据集：同时包含干净样本和对抗样本 (用于Madry's AT)
    
    在__getitem__中生成对抗样本，然后混合返回
    适用于标准对抗训练（非动态生成模式）
    """
    
    def __init__(self, clean_dataset, model, attack_method='pgd',
                 eps=0.05, steps=10, alpha=None, device='cuda'):
        self.clean_data = clean_dataset
        self.model = model
        self.device = device
        
        if alpha is None:
            alpha = eps / 4
            
        # 训练时使用较少步数（10步），防止过拟合
        if attack_method == 'pgd':
            self.attacker = PGD(model, device=device, eps=eps, num_steps=steps, alpha=alpha, random_start=True)
        elif attack_method == 'sap':
            self.attacker = SAP(model, device=device, eps=eps, num_steps=steps, use_pgd_init=True)
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
    
    def __len__(self):
        return len(self.clean_data)
    
    def __getitem__(self, idx):
        """
        返回干净样本（生成对抗样本在训练循环中进行）
        
        Returns:
            x: 干净样本
            y: 标签
        """
        x, y = self.clean_data[idx]
        return x, y
    
    def generate_adversarial(self, x, y):
        """
        批量生成对抗样本
        
        Args:
            x: [B, C, L] 干净样本
            y: [B] 标签
            
        Returns:
            x_adv: [B, C, L] 对抗样本
        """
        x = x.to(self.device)
        y = y.to(self.device)
        return self.attacker.generate(x, y)


def test_adversarial_dataset():
    """测试AdversarialDataset"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.ecg_cnn import ECG_CNN
    from data.mitbih_loader import MITBIHDataset
    from torch.utils.data import DataLoader
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = ECG_CNN(num_classes=5).to(device)
    model.eval()
    
    # 创建模拟数据集用于测试
    import tempfile
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    n_samples = 100
    mock_data = np.random.randn(n_samples, 187).astype(np.float32)
    mock_labels = np.random.randint(0, 5, n_samples)
    mock_full = np.column_stack([mock_data, mock_labels])
    
    temp_path = tempfile.mktemp(suffix='.csv')
    np.savetxt(temp_path, mock_full, delimiter=',')
    
    try:
        # 测试PGD对抗数据集
        print("\n" + "="*50)
        print("Testing AdversarialDataset with PGD attack")
        print("="*50)
        
        clean_dataset = MITBIHDataset(temp_path, transform='normalize', preload=True)
        adv_dataset = AdversarialDataset(
            clean_dataset=clean_dataset,
            model=model,
            attack_method='pgd',
            eps=0.05,
            steps=10,  # 测试用10步
            device=device
        )
        
        # 测试DataLoader
        loader = DataLoader(adv_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        print("\nFetching one batch...")
        x_clean, x_adv, y = next(iter(loader))
        
        print(f"x_clean shape: {x_clean.shape}")
        print(f"x_adv shape: {x_adv.shape}")
        print(f"y shape: {y.shape}")
        print(f"Perturbation max: {(x_adv - x_clean).abs().max().item():.6f}")
        print(f"Epsilon: {adv_dataset.eps}")
        
        # 验证对抗样本有效性
        with torch.no_grad():
            out_clean = model(x_clean.to(device)).argmax(dim=1)
            out_adv = model(x_adv.to(device)).argmax(dim=1)
        
        print(f"\nOriginal predictions: {out_clean[:8]}")
        print(f"Adversarial predictions: {out_adv[:8]}")
        attack_success = (out_clean != out_adv).sum().item()
        print(f"Attack success: {attack_success}/{len(y)} samples")
        
        # 测试SAP对抗数据集
        print("\n" + "="*50)
        print("Testing AdversarialDataset with SAP attack")
        print("="*50)
        
        adv_dataset_sap = AdversarialDataset(
            clean_dataset=clean_dataset,
            model=model,
            attack_method='sap',
            eps=0.05,
            steps=10,
            device=device
        )
        
        loader_sap = DataLoader(adv_dataset_sap, batch_size=8, shuffle=False, num_workers=0)
        x_clean_sap, x_adv_sap, y_sap = next(iter(loader_sap))
        
        print(f"x_clean shape: {x_clean_sap.shape}")
        print(f"x_adv shape: {x_adv_sap.shape}")
        print(f"Perturbation max: {(x_adv_sap - x_clean_sap).abs().max().item():.6f}")
        
        print("\n✅ AdversarialDataset test passed!")
        
    finally:
        os.remove(temp_path)


if __name__ == "__main__":
    test_adversarial_dataset()
