"""
提取 MIT-BIH 训练集的手工特征
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from features.ecg_features import ECGFeatureExtractor


def main():
    print("=" * 60)
    print("提取 MIT-BIH 训练集手工特征")
    print("=" * 60)
    
    # 加载训练数据
    train_df = pd.read_csv('data/mitbih_train.csv', header=None)
    X_train = train_df.iloc[:, :-1].values.reshape(-1, 1, 187).astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(int)
    
    print(f"训练集: X={X_train.shape}, y={y_train.shape}")
    
    # 创建特征提取器
    extractor = ECGFeatureExtractor(sampling_rate=360)
    
    # 提取特征
    print("\n开始提取训练集特征...")
    features = extractor.extract_batch(X_train.squeeze(1), verbose=True)
    
    print(f"\n特征矩阵形状: {features.shape}")
    
    # 保存
    output_path = 'data/handcrafted_features_train.npy'
    np.save(output_path, features)
    print(f"\n特征已保存: {output_path}")
    
    # 统计信息
    print("\n特征统计:")
    feature_names = extractor.feature_names
    for i, name in enumerate(feature_names):
        feat_values = features[:, i]
        non_zero = np.sum(feat_values != 0)
        print(f"  {name:20s}: 均值={np.mean(feat_values):8.4f}, "
              f"标准差={np.std(feat_values):8.4f}, "
              f"非零比例={non_zero/len(feat_values)*100:.1f}%")


if __name__ == "__main__":
    main()
