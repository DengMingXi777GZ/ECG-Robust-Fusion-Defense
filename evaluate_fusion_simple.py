"""
简化版融合模型评估
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.fusion_model import DualBranchECG
from data.fusion_dataset import FusionDataset
from attacks.pgd import PGD
from attacks.sap import SAP

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载测试数据
print("\n加载测试数据...")
test_df = pd.read_csv('data/mitbih_test.csv', header=None)
X_test = test_df.iloc[:, :-1].values.reshape(-1, 1, 187).astype(np.float32)
y_test = test_df.iloc[:, -1].values.astype(np.int64)

# 加载手工特征
hc_features = np.load('data/handcrafted_features_test.npy')

# 标准化手工特征 (与训练时一致)
hc_mean = np.mean(hc_features, axis=0)
hc_std = np.std(hc_features, axis=0)
hc_std = np.where(hc_std == 0, 1.0, hc_std)
hc_features_norm = (hc_features - hc_mean) / hc_std

print(f"测试集: X={X_test.shape}, y={y_test.shape}")
print(f"手工特征: {hc_features_norm.shape}")

# 创建数据集
from torch.utils.data import TensorDataset, DataLoader
test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(hc_features_norm),
    torch.from_numpy(y_test)
)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 加载融合模型
print("\n加载融合模型...")
model = DualBranchECG(num_classes=5, pretrained_path=None, freeze_deep_branch=False)
checkpoint = torch.load('checkpoints/fusion_best.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 评估 Clean 准确率
print("\n评估 Clean 准确率...")
correct = 0
total = 0
with torch.no_grad():
    for signals, features, labels in tqdm(test_loader):
        signals = signals.to(device)
        features = features.to(device)
        labels = labels.to(device)
        
        outputs, _, _ = model(signals, features)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

clean_acc = 100. * correct / total
print(f"Clean Accuracy: {clean_acc:.2f}%")

# 评估 PGD-20 鲁棒性
print("\n评估 PGD-20 鲁棒性...")

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, handcrafted_features, device):
        super().__init__()
        self.model = model
        self.handcrafted = torch.from_numpy(handcrafted_features).float().to(device)
        self.device = device
        self.idx = 0
    
    def forward(self, x):
        batch_size = x.size(0)
        # 获取对应的手工特征
        start_idx = self.idx
        end_idx = min(start_idx + batch_size, len(self.handcrafted))
        hc = self.handcrafted[start_idx:end_idx]
        
        # 如果批次大小不匹配，循环使用
        if len(hc) < batch_size:
            hc = torch.cat([hc, self.handcrafted[:batch_size - len(hc)]], dim=0)
        
        outputs, _, _ = self.model(x, hc)
        
        # 更新索引
        self.idx = (self.idx + batch_size) % len(self.handcrafted)
        return outputs

wrapped_model = ModelWrapper(model, hc_features_norm, device)
wrapped_model.eval()

# 创建 PGD 攻击器
pgd_attack = PGD(model=wrapped_model, device=device, eps=0.01, num_steps=20)

correct = 0
total = 0
wrapped_model.idx = 0  # 重置索引

for signals, features, labels in tqdm(test_loader):
    signals = signals.to(device)
    labels = labels.to(device)
    
    # 生成对抗样本
    signals_adv = pgd_attack.generate(signals, labels)
    
    # 评估
    with torch.no_grad():
        outputs, _, _ = model(signals_adv, features.to(device))
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

pgd_acc = 100. * correct / total
print(f"PGD-20 Robust Accuracy: {pgd_acc:.2f}%")

# 保存结果
print("\n" + "="*60)
print("融合模型评估结果")
print("="*60)
print(f"Clean Accuracy: {clean_acc:.2f}%")
print(f"PGD-20 Robust Accuracy: {pgd_acc:.2f}%")
print("="*60)

results = {
    'clean_acc': clean_acc,
    'pgd20_acc': pgd_acc,
}

import json
with open('results/fusion_simple_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n结果已保存: results/fusion_simple_eval.json")
