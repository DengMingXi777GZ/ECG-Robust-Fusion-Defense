"""
对抗样本检测器 (Adversarial Sample Detector)
Layer 3 特征融合层 - 利用 Deep Features 与 Handcrafted Features 的不一致性检测对抗样本

核心原理:
    - 正常样本: Deep CNN 和人工特征给出一致的预测
    - 对抗样本: Deep CNN 被骗，但人工特征（基于生理规则）可能仍正确，产生分歧
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import os

# 处理导入问题
try:
    from models.fusion_model import DualBranchECG
except ImportError:
    from fusion_model import DualBranchECG


class AdversarialDetector(nn.Module):
    """
    对抗样本检测器
    
    通过比较 Deep Branch 和 Handcrafted Branch 的预测差异来检测对抗样本。
    正常样本的两个分支预测一致，对抗样本则会产生分歧。
    
    Attributes:
        fusion_model: 预训练的双分支融合模型
        aux_deep_classifier: Deep branch 辅助分类器 -> 5 classes
        aux_hc_classifier: Handcrafted branch 辅助分类器 -> 5 classes
        detector: 检测头，输出对抗概率
    """
    
    def __init__(self, fusion_model):
        """
        初始化对抗样本检测器
        
        Args:
            fusion_model: DualBranchECG 实例，用于提取特征
        """
        super().__init__()
        self.fusion_model = fusion_model
        
        # 冻结融合模型参数
        for param in self.fusion_model.parameters():
            param.requires_grad = False
        
        # 辅助分类器：将特征映射到类别 logits
        self.aux_deep_classifier = nn.Linear(128, 5)  # Deep branch -> 5 classes
        self.aux_hc_classifier = nn.Linear(16, 5)     # Handcrafted branch -> 5 classes
        
        # 检测头：输入是两个分支的 logits 差异
        # 输入维度 = Deep_logits [5] + HC_logits [5] + disagreement [5] = 15
        self.detector = nn.Sequential(
            nn.Linear(5 * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出概率：0=Clean, 1=Adversarial
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化检测器和辅助分类器的权重"""
        for m in self.aux_deep_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.aux_hc_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.detector.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x_signal, x_handcrafted):
        """
        前向传播
        
        Args:
            x_signal: [B, 1, 187] 原始 ECG 信号
            x_handcrafted: [B, 12] 手工特征
        
        Returns:
            detection_prob: [B, 1] 对抗概率，0=Clean, 1=Adversarial
            deep_logits: [B, 5] Deep branch 的分类 logits
            hc_logits: [B, 5] Handcrafted branch 的分类 logits
            disagreement: [B, 5] 两个分支的预测差异
        """
        # 1. 通过 fusion_model 获取 deep_feat 和 hc_feat
        with torch.no_grad():
            _, deep_feat, hc_feat = self.fusion_model(x_signal, x_handcrafted)
        
        # 2. 用辅助分类器获取各自 logits
        deep_logits = self.aux_deep_classifier(deep_feat)  # [B, 5]
        hc_logits = self.aux_hc_classifier(hc_feat)        # [B, 5]
        
        # 3. 计算 disagreement = |deep_logits - hc_logits|
        disagreement = torch.abs(deep_logits - hc_logits)  # [B, 5]
        
        # 4. 拼接特征并输入检测器
        detector_input = torch.cat([deep_logits, hc_logits, disagreement], dim=1)  # [B, 15]
        detection_prob = self.detector(detector_input)  # [B, 1]
        
        return detection_prob, deep_logits, hc_logits, disagreement
    
    def predict(self, x_signal, x_handcrafted, threshold=0.5):
        """
        预测样本是否为对抗样本
        
        Args:
            x_signal: [B, 1, 187] 原始 ECG 信号
            x_handcrafted: [B, 12] 手工特征
            threshold: 判定阈值，默认 0.5
        
        Returns:
            is_adversarial: [B] 布尔张量，True 表示对抗样本
            detection_prob: [B, 1] 对抗概率
        """
        detection_prob, _, _, _ = self.forward(x_signal, x_handcrafted)
        is_adversarial = (detection_prob > threshold).squeeze()
        return is_adversarial, detection_prob
    
    def count_trainable_parameters(self):
        """计算可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def prepare_detection_data(clean_ratio=0.5, batch_size=32, num_workers=0):
    """
    准备检测器的训练数据 (Clean:Adv = 1:1)
    
    从 MIT-BIH 测试集加载正常数据，从对抗样本文件加载对抗样本。
    
    Args:
        clean_ratio: 正常样本比例，默认 0.5 (即 Clean:Adv = 1:1)
        batch_size: 批次大小
        num_workers: 数据加载线程数
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    print("[Info] 准备检测器训练数据...")
    
    # 加载 MIT-BIH 测试集 (Clean 样本)
    import pandas as pd
    test_df = pd.read_csv('data/mitbih_test.csv', header=None)
    X_test = test_df.iloc[:, :-1].values.reshape(-1, 1, 187).astype(np.float32)
    
    # 加载手工特征
    if os.path.exists('data/handcrafted_features_test.npy'):
        handcrafted_features = np.load('data/handcrafted_features_test.npy')
    else:
        print("[Warning] 手工特征不存在，使用零向量")
        handcrafted_features = np.zeros((len(X_test), 12), dtype=np.float32)
    
    clean_signals = torch.from_numpy(X_test)
    clean_handcrafted = torch.from_numpy(handcrafted_features)
    clean_labels = torch.zeros(len(clean_signals))  # 0 = Clean
    
    print(f"[Info] Clean 样本数量: {len(clean_signals)}")
    
    # 加载对抗样本
    adv_path = 'data/adversarial/eps005/test_pgd.pt'
    if not os.path.exists(adv_path):
        adv_path = 'data/adversarial/test_pgd.pt'
    
    if os.path.exists(adv_path):
        adv_data = torch.load(adv_path, map_location='cpu', weights_only=False)
        adv_signals = adv_data['x_adv']
        # 使用相同的预提取手工特征作为对抗样本的特征
        num_adv = min(len(adv_signals), len(handcrafted_features))
        adv_handcrafted = torch.from_numpy(handcrafted_features[:num_adv])
        adv_signals = adv_signals[:num_adv]
        adv_labels = torch.ones(len(adv_signals))  # 1 = Adversarial
        print(f"[Info] Adversarial 样本数量: {len(adv_signals)}")
    else:
        print(f"[Warning] 找不到对抗样本文件: {adv_path}")
        print("[Warning] 使用模拟对抗样本数据")
        # 创建模拟对抗样本（添加噪声）
        num_adv = min(len(clean_signals), 1000)
        indices = torch.randperm(len(clean_signals))[:num_adv]
        adv_signals = clean_signals[indices] + torch.randn_like(clean_signals[indices]) * 0.1
        adv_handcrafted = clean_handcrafted[indices] + torch.randn_like(clean_handcrafted[indices]) * 0.1
        adv_labels = torch.ones(len(adv_signals))
    
    # 平衡数据集 (Clean:Adv = 1:1)
    num_samples = min(len(clean_signals), len(adv_signals))
    
    # 随机采样
    clean_indices = torch.randperm(len(clean_signals))[:num_samples]
    adv_indices = torch.randperm(len(adv_signals))[:num_samples]
    
    clean_signals = clean_signals[clean_indices]
    clean_handcrafted = clean_handcrafted[clean_indices]
    clean_labels = clean_labels[clean_indices]
    
    adv_signals = adv_signals[adv_indices]
    adv_handcrafted = adv_handcrafted[adv_indices]
    adv_labels = adv_labels[adv_indices]
    
    # 合并数据
    all_signals = torch.cat([clean_signals, adv_signals], dim=0)
    all_handcrafted = torch.cat([clean_handcrafted, adv_handcrafted], dim=0)
    all_labels = torch.cat([clean_labels, adv_labels], dim=0)
    
    # 打乱数据
    perm = torch.randperm(len(all_signals))
    all_signals = all_signals[perm]
    all_handcrafted = all_handcrafted[perm]
    all_labels = all_labels[perm]
    
    # 划分训练集和验证集 (8:2)
    split_idx = int(0.8 * len(all_signals))
    
    train_signals = all_signals[:split_idx]
    train_handcrafted = all_handcrafted[:split_idx]
    train_labels = all_labels[:split_idx]
    
    val_signals = all_signals[split_idx:]
    val_handcrafted = all_handcrafted[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"[Info] 训练集大小: {len(train_signals)} (Clean: {(train_labels==0).sum().item()}, Adv: {(train_labels==1).sum().item()})")
    print(f"[Info] 验证集大小: {len(val_signals)} (Clean: {(val_labels==0).sum().item()}, Adv: {(val_labels==1).sum().item()})")
    
    # 创建 DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(train_signals, train_handcrafted, train_labels)
    val_dataset = TensorDataset(val_signals, val_handcrafted, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 验证集也作为测试集使用
    return train_loader, val_loader, val_loader


def train_detector(detector, train_loader, val_loader, num_epochs=20, lr=0.001, device='cpu'):
    """
    训练对抗样本检测器
    
    Args:
        detector: AdversarialDetector 实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        device: 训练设备
    
    Returns:
        history: 训练历史记录
    """
    detector = detector.to(device)
    
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_acc': []
    }
    
    best_auc = 0.0
    best_state = None
    
    print("\n" + "=" * 60)
    print("开始训练对抗样本检测器")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # 训练阶段
        detector.train()
        train_loss = 0.0
        
        for signals, handcrafted, labels in train_loader:
            signals = signals.to(device)
            handcrafted = handcrafted.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            detection_prob, _, _, _ = detector(signals, handcrafted)
            loss = criterion(detection_prob, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        detector.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for signals, handcrafted, labels in val_loader:
                signals = signals.to(device)
                handcrafted = handcrafted.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                detection_prob, _, _, _ = detector(signals, handcrafted)
                loss = criterion(detection_prob, labels)
                
                val_loss += loss.item()
                all_probs.append(detection_prob.cpu())
                all_labels.append(labels.cpu())
        
        val_loss /= len(val_loader)
        
        # 计算指标
        all_probs = torch.cat(all_probs).numpy().flatten()
        all_labels = torch.cat(all_labels).numpy().flatten()
        
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except:
            val_auc = 0.5
        
        val_acc = accuracy_score(all_labels, (all_probs > 0.5).astype(int))
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = detector.state_dict().copy()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val Acc: {val_acc:.4f}")
    
    # 加载最佳模型
    if best_state is not None:
        detector.load_state_dict(best_state)
        print(f"\n[Info] 加载最佳模型 (AUC={best_auc:.4f})")
    
    print("=" * 60)
    
    return history


def evaluate_detector(detector, test_loader, device='cpu'):
    """
    评估检测器性能
    
    Args:
        detector: AdversarialDetector 实例
        test_loader: 测试数据加载器
        device: 评估设备
    
    Returns:
        metrics: 评估指标字典
    """
    detector.eval()
    detector = detector.to(device)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for signals, handcrafted, labels in test_loader:
            signals = signals.to(device)
            handcrafted = handcrafted.to(device)
            
            detection_prob, _, _, _ = detector(signals, handcrafted)
            all_probs.append(detection_prob.cpu())
            all_labels.append(labels.cpu())
    
    all_probs = torch.cat(all_probs).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()
    
    # 计算指标
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    preds = (all_probs > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    
    # 计算 PGD 样本检出率
    adv_mask = all_labels == 1
    if adv_mask.sum() > 0:
        pgd_detect_rate = (preds[adv_mask] == 1).sum() / adv_mask.sum()
    else:
        pgd_detect_rate = 0.0
    
    metrics = {
        'auc': auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'pgd_detect_rate': pgd_detect_rate
    }
    
    return metrics


def test_detector():
    """
    测试对抗样本检测器
    
    测试内容包括:
        1. 模型结构测试
        2. 前向传播测试
        3. 训练流程测试
        4. 评估指标测试
    """
    print("=" * 60)
    print("对抗样本检测器测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] 使用设备: {device}")
    
    # 1. 创建融合模型
    print("\n[1/4] 创建融合模型...")
    fusion_model = DualBranchECG(
        num_classes=5,
        pretrained_path='checkpoints/at_nsr.pth',
        freeze_deep_branch=False
    )
    fusion_model.eval()
    
    # 2. 创建检测器
    print("\n[2/4] 创建检测器...")
    detector = AdversarialDetector(fusion_model)
    
    # 检查可训练参数
    trainable_params = detector.count_trainable_parameters()
    print(f"[Info] 检测器可训练参数量: {trainable_params:,}")
    
    # 3. 测试前向传播
    print("\n[3/4] 测试前向传播...")
    batch_size = 8
    x_signal = torch.randn(batch_size, 1, 187)
    x_handcrafted = torch.randn(batch_size, 12)
    
    detection_prob, deep_logits, hc_logits, disagreement = detector(x_signal, x_handcrafted)
    
    print(f"  输入信号: {x_signal.shape}")
    print(f"  输入手工特征: {x_handcrafted.shape}")
    print(f"  检测概率: {detection_prob.shape} (范围: [{detection_prob.min():.3f}, {detection_prob.max():.3f}])")
    print(f"  Deep logits: {deep_logits.shape}")
    print(f"  HC logits: {hc_logits.shape}")
    print(f"  Disagreement: {disagreement.shape}")
    
    # 验证输出形状
    assert detection_prob.shape == (batch_size, 1), f"检测概率形状错误: {detection_prob.shape}"
    assert deep_logits.shape == (batch_size, 5), f"Deep logits 形状错误: {deep_logits.shape}"
    assert hc_logits.shape == (batch_size, 5), f"HC logits 形状错误: {hc_logits.shape}"
    assert disagreement.shape == (batch_size, 5), f"Disagreement 形状错误: {disagreement.shape}"
    
    # 4. 测试预测函数
    print("\n[4/4] 测试预测函数...")
    is_adversarial, prob = detector.predict(x_signal, x_handcrafted)
    print(f"  预测结果: {is_adversarial.shape}")
    print(f"  检测为对抗样本: {is_adversarial.sum().item()}/{batch_size}")
    
    # 5. 简单训练测试
    print("\n[Bonus] 简单训练流程测试...")
    # 创建模拟数据
    from torch.utils.data import TensorDataset, DataLoader
    
    # Clean 样本 (label=0)
    clean_signals = torch.randn(50, 1, 187)
    clean_handcrafted = torch.randn(50, 12)
    clean_labels = torch.zeros(50)
    
    # Adversarial 样本 (label=1)
    adv_signals = torch.randn(50, 1, 187) * 1.5  # 更大的噪声
    adv_handcrafted = torch.randn(50, 12) * 1.5
    adv_labels = torch.ones(50)
    
    all_signals = torch.cat([clean_signals, adv_signals], dim=0)
    all_handcrafted = torch.cat([clean_handcrafted, adv_handcrafted], dim=0)
    all_labels = torch.cat([clean_labels, adv_labels], dim=0)
    
    dataset = TensorDataset(all_signals, all_handcrafted, all_labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 训练几个 epoch
    history = train_detector(
        detector=detector,
        train_loader=train_loader,
        val_loader=train_loader,
        num_epochs=3,
        lr=0.01,
        device=device
    )
    
    # 评估
    metrics = evaluate_detector(detector, train_loader, device=device)
    
    print("\n" + "=" * 60)
    print("检测结果指标:")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  PGD 检出率: {metrics['pgd_detect_rate']:.4f}")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("[OK] 检测器测试通过!")
    print("=" * 60)
    
    return detector, history, metrics


def main():
    """
    主函数：训练对抗样本检测器
    
    使用真实的 MIT-BIH 数据训练检测器
    """
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("=" * 60)
    print("对抗样本检测器训练")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] 使用设备: {device}")
    
    # 1. 创建融合模型
    print("\n[1/5] 加载融合模型...")
    fusion_model = DualBranchECG(
        num_classes=5,
        pretrained_path='checkpoints/fusion_best.pth',
        freeze_deep_branch=False
    )
    fusion_model.eval()
    
    # 2. 创建检测器
    print("\n[2/5] 创建检测器...")
    detector = AdversarialDetector(fusion_model)
    trainable_params = detector.count_trainable_parameters()
    print(f"[Info] 检测器可训练参数量: {trainable_params:,}")
    
    # 3. 准备数据
    print("\n[3/5] 准备训练数据...")
    train_loader, val_loader, test_loader = prepare_detection_data(
        clean_ratio=0.5,
        batch_size=128,
        num_workers=0
    )
    
    # 4. 训练检测器
    print("\n[4/5] 训练检测器...")
    history = train_detector(
        detector=detector,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        lr=0.001,
        device=device
    )
    
    # 5. 评估
    print("\n[5/5] 评估检测器...")
    metrics = evaluate_detector(detector, test_loader, device=device)
    
    print("\n" + "=" * 60)
    print("最终评估结果:")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    if 'f1' in metrics:
        print(f"  F1-Score: {metrics['f1']:.4f}")
    if 'pgd_detect_rate' in metrics:
        print(f"  PGD 检出率: {metrics['pgd_detect_rate']:.2f}%")
    print("=" * 60)
    
    # 6. 保存模型
    save_path = 'checkpoints/detector_best.pth'
    torch.save({
        'model_state_dict': detector.state_dict(),
        'metrics': metrics,
        'history': history
    }, save_path)
    print(f"\n[OK] 检测器已保存: {save_path}")
    
    return detector, history, metrics


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        detector, history, metrics = test_detector()
    else:
        detector, history, metrics = main()
