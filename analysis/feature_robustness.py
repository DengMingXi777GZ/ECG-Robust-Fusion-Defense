"""
对抗样本的特征不变性分析
验证手工特征对对抗扰动的鲁棒性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from features.ecg_features import ECGFeatureExtractor


def load_data():
    """加载干净样本和对抗样本"""
    print("=" * 60)
    print("加载数据")
    print("=" * 60)
    
    # 加载干净测试集
    test_df = pd.read_csv('data/mitbih_test.csv', header=None)
    X_clean = test_df.iloc[:, :-1].values.reshape(-1, 1, 187).astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(int)
    
    # 加载对抗样本 (PGD)
    adv_pgd = torch.load('data/adversarial/eps005/test_pgd.pt', weights_only=False)
    X_pgd = adv_pgd['x_adv'].numpy()
    
    # 加载对抗样本 (SAP)
    adv_sap = torch.load('data/adversarial/eps005/test_sap.pt', weights_only=False)
    X_sap = adv_sap['x_adv'].numpy()
    
    print(f"干净样本: {X_clean.shape}")
    print(f"PGD 对抗样本: {X_pgd.shape}")
    print(f"SAP 对抗样本: {X_sap.shape}")
    
    return X_clean, X_pgd, X_sap, y_test


def extract_features_batch(X, extractor, batch_size=1000, desc="Extracting"):
    """批量提取特征"""
    features = []
    n_samples = len(X)
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = X[i:end_idx]
        batch_features = extractor.extract_batch(batch.squeeze(1), verbose=False)
        features.append(batch_features)
        
        if (i // batch_size + 1) % 5 == 0 or end_idx == n_samples:
            print(f"  {desc}: {end_idx}/{n_samples} ({end_idx/n_samples*100:.1f}%)")
    
    return np.vstack(features)


def compute_feature_drift(X_clean, X_adv, feature_names):
    """
    计算特征漂移（Feature Drift）
    
    Args:
        X_clean: 干净样本特征 [N, 12]
        X_adv: 对抗样本特征 [N, 12]
    
    Returns:
        drift: 每个特征的漂移量 [12]
        relative_drift: 相对漂移量
    """
    # 绝对漂移
    drift = np.mean(np.abs(X_clean - X_adv), axis=0)
    
    # 相对漂移（相对于干净特征的均值）
    clean_mean = np.mean(np.abs(X_clean), axis=0)
    relative_drift = drift / (clean_mean + 1e-6)
    
    return drift, relative_drift


def compute_feature_correlation(X_clean, X_adv):
    """计算特征间的相关系数"""
    correlations = []
    for i in range(X_clean.shape[1]):
        corr = np.corrcoef(X_clean[:, i], X_adv[:, i])[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0.0)
    return np.array(correlations)


def analyze_robustness():
    """主分析函数"""
    print("\n" + "=" * 60)
    print("特征鲁棒性分析")
    print("=" * 60)
    
    # 加载数据
    X_clean, X_pgd, X_sap, y_test = load_data()
    
    # 创建特征提取器
    extractor = ECGFeatureExtractor(sampling_rate=360)
    
    # 检查是否已有预提取的特征
    if os.path.exists('data/handcrafted_features_test.npy'):
        print("\n加载预提取的干净样本特征...")
        F_clean = np.load('data/handcrafted_features_test.npy')
    else:
        # 提取干净样本特征
        print("\n提取干净样本特征...")
        F_clean = extract_features_batch(X_clean, extractor, desc="Clean")
        np.save('data/handcrafted_features_test.npy', F_clean)
    
    # 提取对抗样本特征
    print("\n提取 PGD 对抗样本特征...")
    F_pgd = extract_features_batch(X_pgd, extractor, desc="PGD")
    
    print("\n提取 SAP 对抗样本特征...")
    F_sap = extract_features_batch(X_sap, extractor, desc="SAP")
    
    # 计算特征漂移
    print("\n" + "=" * 60)
    print("特征漂移分析 (PGD)")
    print("=" * 60)
    
    drift_pgd, rel_drift_pgd = compute_feature_drift(F_clean, F_pgd, extractor.feature_names)
    corr_pgd = compute_feature_correlation(F_clean, F_pgd)
    
    print(f"\n{'Feature':<20} {'Drift':>10} {'Rel.Drift':>12} {'Correlation':>12}")
    print("-" * 60)
    for i, name in enumerate(extractor.feature_names):
        print(f"{name:<20} {drift_pgd[i]:>10.4f} {rel_drift_pgd[i]:>12.4f} {corr_pgd[i]:>12.4f}")
    
    # SAP 分析
    print("\n" + "=" * 60)
    print("特征漂移分析 (SAP)")
    print("=" * 60)
    
    drift_sap, rel_drift_sap = compute_feature_drift(F_clean, F_sap, extractor.feature_names)
    corr_sap = compute_feature_correlation(F_clean, F_sap)
    
    print(f"\n{'Feature':<20} {'Drift':>10} {'Rel.Drift':>12} {'Correlation':>12}")
    print("-" * 60)
    for i, name in enumerate(extractor.feature_names):
        print(f"{name:<20} {drift_sap[i]:>10.4f} {rel_drift_sap[i]:>12.4f} {corr_sap[i]:>12.4f}")
    
    # 找出最稳定的特征
    print("\n" + "=" * 60)
    print("最稳定的特征 (漂移 < 0.1)")
    print("=" * 60)
    
    stable_threshold = 0.1
    stable_pgd = [(name, drift) for name, drift in zip(extractor.feature_names, drift_pgd) if drift < stable_threshold]
    stable_sap = [(name, drift) for name, drift in zip(extractor.feature_names, drift_sap) if drift < stable_threshold]
    
    print("\nPGD 攻击下稳定的特征:")
    for name, drift in sorted(stable_pgd, key=lambda x: x[1]):
        print(f"  {name}: {drift:.4f}")
    
    print("\nSAP 攻击下稳定的特征:")
    for name, drift in sorted(stable_sap, key=lambda x: x[1]):
        print(f"  {name}: {drift:.4f}")
    
    # 保存分析结果
    results = {
        'feature_names': extractor.feature_names,
        'drift_pgd': drift_pgd.tolist(),
        'drift_sap': drift_sap.tolist(),
        'relative_drift_pgd': rel_drift_pgd.tolist(),
        'relative_drift_sap': rel_drift_sap.tolist(),
        'correlation_pgd': corr_pgd.tolist(),
        'correlation_sap': corr_sap.tolist()
    }
    
    import json
    with open('results/feature_robustness_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n分析结果已保存: results/feature_robustness_analysis.json")
    
    # 可视化
    plot_feature_drift(extractor.feature_names, drift_pgd, drift_sap, rel_drift_pgd, rel_drift_sap)
    
    return results


def plot_feature_drift(feature_names, drift_pgd, drift_sap, rel_drift_pgd, rel_drift_sap):
    """绘制特征漂移对比图"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        x = np.arange(len(feature_names))
        width = 0.35
        
        # 绝对漂移
        ax = axes[0, 0]
        ax.bar(x - width/2, drift_pgd, width, label='PGD', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, drift_sap, width, label='SAP', color='#3498db', alpha=0.8)
        ax.set_ylabel('Mean Absolute Drift')
        ax.set_title('Feature Drift: Absolute')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Threshold')
        ax.grid(axis='y', alpha=0.3)
        
        # 相对漂移
        ax = axes[0, 1]
        ax.bar(x - width/2, rel_drift_pgd, width, label='PGD', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, rel_drift_sap, width, label='SAP', color='#3498db', alpha=0.8)
        ax.set_ylabel('Relative Drift')
        ax.set_title('Feature Drift: Relative')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 漂移对比 - 箱线图
        ax = axes[1, 0]
        drift_data = [drift_pgd, drift_sap]
        bp = ax.boxplot(drift_data, labels=['PGD', 'SAP'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][1].set_facecolor('#3498db')
        ax.set_ylabel('Feature Drift')
        ax.set_title('Feature Drift Distribution')
        ax.grid(axis='y', alpha=0.3)
        
        # 各特征漂移排名
        ax = axes[1, 1]
        drift_sum = drift_pgd + drift_sap
        sorted_idx = np.argsort(drift_sum)
        colors = ['green' if d < 0.2 else 'orange' if d < 0.5 else 'red' for d in drift_sum[sorted_idx]]
        ax.barh(range(len(feature_names)), drift_sum[sorted_idx], color=colors, alpha=0.7)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
        ax.set_xlabel('Total Drift (PGD + SAP)')
        ax.set_title('Feature Stability Ranking')
        ax.axvline(x=0.2, color='green', linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/feature_robustness_analysis.png', dpi=150, bbox_inches='tight')
        print("可视化已保存: results/feature_robustness_analysis.png")
        plt.close()
    except Exception as e:
        print(f"可视化失败: {e}")


if __name__ == "__main__":
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    analyze_robustness()
