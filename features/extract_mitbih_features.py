"""
提取 MIT-BIH 数据集的手工特征
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from features.ecg_features import ECGFeatureExtractor


def load_mitbih_data():
    """加载 MIT-BIH 数据集"""
    data_dir = 'data'
    
    # 加载测试集
    test_df = pd.read_csv(os.path.join(data_dir, 'mitbih_test.csv'), header=None)
    
    # 分离特征和标签
    X_test = test_df.iloc[:, :-1].values  # [N, 187]
    y_test = test_df.iloc[:, -1].values.astype(int)  # [N]
    
    # 调整形状为 [N, 1, 187]
    X_test = X_test.reshape(-1, 1, 187).astype(np.float32)
    
    print(f"测试集: X={X_test.shape}, y={y_test.shape}")
    
    return X_test, y_test


def extract_and_save_features():
    """提取并保存特征"""
    print("=" * 60)
    print("MIT-BIH 数据集特征提取")
    print("=" * 60)
    
    # 加载数据
    X_test, y_test = load_mitbih_data()
    
    # 创建特征提取器
    extractor = ECGFeatureExtractor(sampling_rate=360)
    
    # 提取测试集特征
    print("\n开始提取测试集特征...")
    features = extractor.extract_batch(X_test.squeeze(1), verbose=True)
    
    print(f"\n特征矩阵形状: {features.shape}")
    print(f"特征名称: {extractor.feature_names}")
    
    # 特征统计
    print("\n特征统计信息:")
    for i, name in enumerate(extractor.feature_names):
        feat_values = features[:, i]
        non_zero = np.sum(feat_values != 0)
        print(f"  {name:20s}: 均值={np.mean(feat_values):8.4f}, "
              f"标准差={np.std(feat_values):8.4f}, "
              f"非零比例={non_zero/len(feat_values)*100:.1f}%")
    
    # 保存特征
    output_path = 'data/handcrafted_features_test.npy'
    np.save(output_path, features)
    print(f"\n特征已保存: {output_path}")
    
    # 同时保存标签（方便后续使用）
    np.save('data/handcrafted_labels_test.npy', y_test)
    print(f"标签已保存: data/handcrafted_labels_test.npy")
    
    return features, y_test


if __name__ == "__main__":
    extract_and_save_features()
