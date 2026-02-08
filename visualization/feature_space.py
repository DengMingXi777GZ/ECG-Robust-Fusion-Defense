"""
Layer 3 特征空间可视化脚本
用于生成论文的 t-SNE 特征分布图和特征重要性分析图

可视化内容:
    1. t-SNE 特征分布图 (2张): Deep Features (128维) 和 Handcrafted Features (12维)
    2. 特征重要性柱状图 (1张): 使用 Permutation Importance 分析 12 个人工特征
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 设置 matplotlib 后端和 KMP 环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 处理导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.fusion_model import DualBranchECG
    from data.fusion_dataset import FusionDataset, load_mitbih_data_for_fusion
    from features.ecg_features import ECGFeatureExtractor
except ImportError as e:
    print(f"[Error] 导入失败: {e}")
    raise


# ==================== 特征名称定义 ====================
HANDCRAFTED_FEATURE_NAMES = [
    # 心率变异性 (4维)
    'RR_mean', 'RR_std', 'RR_max', 'RR_min',
    # 波形形态 (3维)
    'QRS_width', 'PR_interval', 'QT_interval',
    # 频域特征 (3维)
    'LF_power', 'HF_power', 'LF_HF_ratio',
    # 统计特征 (2维)
    'Signal_skewness', 'Signal_kurtosis'
]


def load_model_and_data(model_path='checkpoints/fusion_best.pth', device='cpu', use_at_nsr_fallback=True):
    """
    加载融合模型和数据
    
    Args:
        model_path: 融合模型权重路径
        device: 计算设备
        use_at_nsr_fallback: 如果融合模型不存在，是否使用 AT+NSR 模型作为回退
        
    Returns:
        model: 加载好的 DualBranchECG 模型
        clean_data: 干净样本数据字典
        pgd_data: PGD 对抗样本数据字典
        sap_data: SAP 对抗样本数据字典
    """
    print("=" * 60)
    print("加载模型和数据")
    print("=" * 60)
    
    # 检查融合模型是否存在
    fusion_model_exists = os.path.exists(model_path)
    
    # 创建融合模型
    # 如果融合模型不存在且允许回退，不加载预训练权重（后续手动加载AT+NSR）
    pretrained_path = 'checkpoints/at_nsr.pth' if (not fusion_model_exists and use_at_nsr_fallback) else 'checkpoints/at_nsr.pth'
    
    model = DualBranchECG(
        num_classes=5,
        pretrained_path=pretrained_path,
        freeze_deep_branch=False
    )
    
    # 加载训练好的融合模型权重（如果存在）
    if fusion_model_exists:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"[OK] 加载融合模型: {model_path}")
        except Exception as e:
            print(f"[Warning] 无法加载 {model_path}: {e}")
            if use_at_nsr_fallback:
                print("[Info] 使用 AT+NSR 模型作为 Deep Branch")
            else:
                print("[Warning] 使用随机初始化的模型")
    else:
        print(f"[Info] 融合模型不存在: {model_path}")
        if use_at_nsr_fallback and os.path.exists('checkpoints/at_nsr.pth'):
            print("[Info] 使用 AT+NSR 模型作为 Deep Branch（特征提取）")
        else:
            print("[Warning] 使用随机初始化的模型（仅用于测试）")
    
    model = model.to(device)
    model.eval()
    
    # 加载 MIT-BIH 测试数据（Clean）
    print("\n[1/3] 加载 Clean 样本...")
    _, test_dataset = load_mitbih_data_for_fusion(data_dir='data', use_precomputed=True)
    
    # 提取 Clean 样本
    clean_signals = test_dataset.signals
    clean_handcrafted = test_dataset.handcrafted
    clean_labels = test_dataset.labels
    
    print(f"  Clean 样本数量: {len(clean_signals)}")
    
    # 加载 PGD 对抗样本
    print("\n[2/3] 加载 PGD 对抗样本...")
    pgd_signals, pgd_handcrafted, pgd_labels = load_adversarial_data('pgd', device)
    print(f"  PGD 样本数量: {len(pgd_signals)}")
    
    # 加载 SAP 对抗样本
    print("\n[3/3] 加载 SAP 对抗样本...")
    sap_signals, sap_handcrafted, sap_labels = load_adversarial_data('sap', device)
    print(f"  SAP 样本数量: {len(sap_signals)}")
    
    # 构建数据字典
    clean_data = {
        'signals': clean_signals,
        'handcrafted': clean_handcrafted,
        'labels': clean_labels,
        'type': 'clean'
    }
    
    pgd_data = {
        'signals': pgd_signals,
        'handcrafted': pgd_handcrafted,
        'labels': pgd_labels,
        'type': 'pgd'
    }
    
    sap_data = {
        'signals': sap_signals,
        'handcrafted': sap_handcrafted,
        'labels': sap_labels,
        'type': 'sap'
    }
    
    print("\n" + "=" * 60)
    print("[OK] 数据加载完成")
    print("=" * 60)
    
    return model, clean_data, pgd_data, sap_data


def load_adversarial_data(attack='pgd', device='cpu'):
    """
    加载对抗样本数据
    
    Args:
        attack: 攻击类型 ('pgd', 'fgsm', 'sap')
        device: 设备
        
    Returns:
        signals: [N, 1, 187] 对抗信号
        handcrafted: [N, 12] 手工特征
        labels: [N] 标签
    """
    # 尝试加载对抗样本
    adv_paths = [
        f'data/adversarial/eps005/test_{attack}.pt',
        f'data/adversarial/test_{attack}.pt'
    ]
    
    adv_data = None
    for path in adv_paths:
        if os.path.exists(path):
            adv_data = torch.load(path, map_location=device, weights_only=False)
            break
    
    if adv_data is None:
        raise FileNotFoundError(f"找不到 {attack} 对抗样本文件")
    
    # 提取信号和标签
    if 'x_adv' in adv_data:
        signals = adv_data['x_adv'].numpy()
        labels = adv_data['y'].numpy().astype(np.int64)
    elif 'adversarial_signals' in adv_data:
        signals = adv_data['adversarial_signals'].numpy()
        labels = adv_data['labels'].numpy().astype(np.int64)
    else:
        raise ValueError(f"无法解析对抗样本数据格式: {adv_data.keys()}")
    
    # 加载或提取手工特征
    features_path = f'data/handcrafted_features_{attack}.npy'
    
    if os.path.exists(features_path):
        handcrafted = np.load(features_path)
    else:
        print(f"  提取 {attack} 对抗样本特征...")
        extractor = ECGFeatureExtractor()
        handcrafted = extractor.extract_batch(signals.squeeze(1), verbose=False)
        np.save(features_path, handcrafted)
    
    return signals, handcrafted, labels


def extract_features_from_model(model, clean_data, pgd_data, sap_data, device='cpu', max_samples=1000):
    """
    从融合模型中提取 Deep Features 和 Handcrafted Features
    
    Args:
        model: DualBranchECG 模型
        clean_data: 干净样本数据
        pgd_data: PGD 对抗样本数据
        sap_data: SAP 对抗样本数据
        device: 设备
        max_samples: 每个类别的最大样本数
        
    Returns:
        deep_features: [N, 128] Deep 特征
        hc_features_original: [N, 12] 原始手工特征
        labels_sample: [N] 样本标签（0=Clean, 1=PGD, 2=SAP）
    """
    print("\n" + "=" * 60)
    print("提取特征")
    print("=" * 60)
    
    model.eval()
    
    # 限制样本数量以平衡各类别
    def subsample(data, max_n):
        n = min(len(data['signals']), max_n)
        indices = np.random.choice(len(data['signals']), n, replace=False)
        return {
            'signals': data['signals'][indices],
            'handcrafted': data['handcrafted'][indices],
            'labels': data['labels'][indices]
        }
    
    clean_sub = subsample(clean_data, max_samples)
    pgd_sub = subsample(pgd_data, max_samples)
    sap_sub = subsample(sap_data, max_samples)
    
    # 合并数据
    all_signals = np.vstack([clean_sub['signals'], pgd_sub['signals'], sap_sub['signals']])
    all_handcrafted = np.vstack([clean_sub['handcrafted'], pgd_sub['handcrafted'], sap_sub['handcrafted']])
    
    # 创建样本类型标签: 0=Clean, 1=PGD, 2=SAP
    sample_labels = np.concatenate([
        np.zeros(len(clean_sub['signals'])),
        np.ones(len(pgd_sub['signals'])),
        np.full(len(sap_sub['signals']), 2)
    ]).astype(np.int64)
    
    print(f"样本分布: Clean={len(clean_sub['signals'])}, PGD={len(pgd_sub['signals'])}, SAP={len(sap_sub['signals'])}")
    
    # 分批提取特征
    batch_size = 64
    deep_features_list = []
    hc_features_list = []
    
    with torch.no_grad():
        for i in range(0, len(all_signals), batch_size):
            batch_signals = torch.from_numpy(all_signals[i:i+batch_size]).float().to(device)
            batch_handcrafted = torch.from_numpy(all_handcrafted[i:i+batch_size]).float().to(device)
            
            # 前向传播获取特征
            _, deep_feat, hc_feat = model(batch_signals, batch_handcrafted)
            
            deep_features_list.append(deep_feat.cpu().numpy())
            hc_features_list.append(hc_feat.cpu().numpy())
    
    deep_features = np.vstack(deep_features_list)
    hc_features = np.vstack(hc_features_list)
    
    print(f"[OK] Deep 特征形状: {deep_features.shape}")
    print(f"[OK] Handcrafted 特征形状: {hc_features.shape}")
    
    return deep_features, hc_features, all_handcrafted, sample_labels


def visualize_tsne(features, labels, title, save_path, perplexity=30):
    """
    使用 t-SNE 可视化特征空间分布
    
    Args:
        features: [N, D] 特征矩阵
        labels: [N] 样本类型标签（0=Clean, 1=PGD, 2=SAP）
        title: 图表标题
        save_path: 保存路径
        perplexity: t-SNE 困惑度参数
    """
    print(f"\n生成 t-SNE 可视化: {title}")
    
    # 限制样本数以加速 t-SNE
    max_tsne_samples = 2000
    if len(features) > max_tsne_samples:
        indices = np.random.choice(len(features), max_tsne_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    # 运行 t-SNE
    print(f"  运行 t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(features)-1))
    features_2d = tsne.fit_transform(features)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # 定义颜色和标签
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # 蓝、红、绿
    type_names = ['Clean', 'PGD', 'SAP']
    markers = ['o', 's', '^']
    
    # 绘制散点图
    for i in range(3):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(
                features_2d[mask, 0], 
                features_2d[mask, 1],
                c=colors[i],
                label=type_names[i],
                alpha=0.6,
                s=30,
                marker=markers[i],
                edgecolors='none'
            )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 保存到: {save_path}")


def compute_permutation_importance_sklearn(model, X, y, feature_names, n_repeats=10, random_state=42):
    """
    使用 sklearn 的 permutation_importance 计算特征重要性
    
    Args:
        model: 训练好的模型（需要 predict 方法）
        X: [N, D] 特征矩阵
        y: [N] 标签
        feature_names: 特征名称列表
        n_repeats: 重复次数
        random_state: 随机种子
        
    Returns:
        importances_mean: 每个特征的平均重要性
        importances_std: 每个特征的重要性标准差
    """
    print(f"\n计算 Permutation Importance (n_repeats={n_repeats})...")
    
    result = permutation_importance(
        model, X, y, 
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='accuracy',
        n_jobs=1
    )
    
    return result.importances_mean, result.importances_std


def compute_manual_permutation_importance(model, X, y, device='cpu', n_repeats=5):
    """
    手动计算 Permutation Importance（适用于 PyTorch 模型）
    
    Args:
        model: PyTorch 模型
        X: [N, D] 特征矩阵 (numpy)
        y: [N] 标签 (numpy)
        device: 设备
        n_repeats: 重复次数
        
    Returns:
        importances: 每个特征的平均重要性
    """
    print(f"\n手动计算 Permutation Importance (n_repeats={n_repeats})...")
    
    model.eval()
    
    # 计算基线准确率
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).long().to(device)
        outputs = model(x_signal=None, x_handcrafted=X_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        preds = torch.argmax(outputs, dim=1)
        baseline_acc = (preds == y_tensor).float().mean().item()
    
    print(f"  基线准确率: {baseline_acc:.4f}")
    
    n_features = X.shape[1]
    importances = np.zeros(n_features)
    
    for feat_idx in range(n_features):
        acc_drops = []
        
        for _ in range(n_repeats):
            # 复制数据并打乱当前特征
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feat_idx])
            
            # 计算打乱后的准确率
            with torch.no_grad():
                X_perm_tensor = torch.from_numpy(X_permuted).float().to(device)
                outputs = model(x_signal=None, x_handcrafted=X_perm_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                preds = torch.argmax(outputs, dim=1)
                permuted_acc = (preds == y_tensor).float().mean().item()
            
            acc_drops.append(baseline_acc - permuted_acc)
        
        importances[feat_idx] = np.mean(acc_drops)
    
    return importances


def plot_feature_importance(importances, feature_names, save_path, title='Feature Importance'):
    """
    绘制特征重要性柱状图
    
    Args:
        importances: [D] 特征重要性数组
        feature_names: 特征名称列表
        save_path: 保存路径
        title: 图表标题
    """
    print(f"\n生成特征重要性图...")
    
    # 按重要性排序
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    
    # 颜色映射（重要性越高颜色越深）
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(importances)))
    
    bars = ax.barh(range(len(importances)), sorted_importances, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('Accuracy Drop (Permutation Importance)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, sorted_importances)):
        ax.text(imp + 0.001, i, f'{imp:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 保存到: {save_path}")
    
    # 打印特征重要性排名
    print("\n特征重要性排名:")
    print("-" * 50)
    for i, (name, imp) in enumerate(zip(sorted_names, sorted_importances), 1):
        print(f"{i:2d}. {name:<20s}: {imp:.4f}")


def train_handcrafted_only_model(hc_features, labels, device='cpu'):
    """
    训练一个仅使用手工特征的分类模型（用于特征重要性分析）
    
    Args:
        hc_features: [N, 12] 手工特征
        labels: [N] 标签
        device: 设备
        
    Returns:
        model: 训练好的模型
        accuracy: 测试集准确率
    """
    from models.fusion_model import HandcraftedOnlyModel
    
    print("\n" + "=" * 60)
    print("训练手工特征分类模型")
    print("=" * 60)
    
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        hc_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 转换为 PyTorch 张量
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    # 创建模型
    model = HandcraftedOnlyModel(num_classes=5)
    model = model.to(device)
    
    # 训练参数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 训练循环
    num_epochs = 20
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_handcrafted=batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 评估
        model.eval()
        with torch.no_grad():
            outputs = model(x_handcrafted=X_test_tensor.to(device))
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y_test_tensor.to(device)).float().mean().item()
        
        if acc > best_acc:
            best_acc = acc
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss/len(train_loader):.4f} Acc: {acc:.4f}")
    
    print(f"\n[OK] 最佳测试准确率: {best_acc:.4f}")
    
    return model, best_acc


def main():
    """
    主流程
    """
    print("\n" + "=" * 60)
    print("Layer 3 特征空间可视化")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    
    # 加载模型和数据
    model, clean_data, pgd_data, sap_data = load_model_and_data(
        model_path='checkpoints/fusion_best.pth',
        device=device,
        use_at_nsr_fallback=True
    )
    
    # 提取特征
    deep_features, hc_features_processed, hc_features_original, sample_labels = extract_features_from_model(
        model, clean_data, pgd_data, sap_data, device=device, max_samples=1000
    )
    
    # ==================== 1. t-SNE 可视化 ====================
    print("\n" + "=" * 60)
    print("1. 生成 t-SNE 可视化")
    print("=" * 60)
    
    # 1.1 Deep Features t-SNE
    visualize_tsne(
        features=deep_features,
        labels=sample_labels,
        title='t-SNE Visualization of Deep Features (128-D)',
        save_path='results/tsne_deep_features.png',
        perplexity=30
    )
    
    # 1.2 Handcrafted Features t-SNE（使用原始特征）
    visualize_tsne(
        features=hc_features_original,
        labels=sample_labels,
        title='t-SNE Visualization of Handcrafted Features (12-D)',
        save_path='results/tsne_handcrafted_features.png',
        perplexity=30
    )
    
    # ==================== 2. 特征重要性分析 ====================
    print("\n" + "=" * 60)
    print("2. 特征重要性分析")
    print("=" * 60)
    
    # 准备数据（使用 Clean + PGD + SAP 的所有样本）
    X = hc_features_original
    y = np.concatenate([clean_data['labels'][:1000], 
                        pgd_data['labels'][:1000], 
                        sap_data['labels'][:1000]])
    
    # 使用随机森林计算特征重要性（更稳定）
    print("\n使用随机森林计算特征重要性...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X, y)
    
    # 计算 Permutation Importance
    importances_mean, importances_std = compute_permutation_importance_sklearn(
        rf_model, X, y, HANDCRAFTED_FEATURE_NAMES, n_repeats=10
    )
    
    # 绘制特征重要性图
    plot_feature_importance(
        importances=importances_mean,
        feature_names=HANDCRAFTED_FEATURE_NAMES,
        save_path='results/feature_importance.png',
        title='Feature Importance (Permutation Importance)'
    )
    
    # ==================== 完成 ====================
    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print("\n输出文件:")
    print("  - results/tsne_deep_features.png")
    print("  - results/tsne_handcrafted_features.png")
    print("  - results/feature_importance.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
