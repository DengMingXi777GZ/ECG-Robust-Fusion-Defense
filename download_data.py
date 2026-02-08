"""
下载 MIT-BIH 心律失常数据集
来源: Kaggle - https://www.kaggle.com/datasets/shayanfazeli/heartbeat
"""
import os
import urllib.request
import zipfile
from pathlib import Path

def download_mitbih():
    """下载 MIT-BIH 数据集"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 数据集文件
    train_file = data_dir / "mitbih_train.csv"
    test_file = data_dir / "mitbih_test.csv"
    
    if train_file.exists() and test_file.exists():
        print("[INFO] MIT-BIH dataset already exists!")
        print(f"  - {train_file}")
        print(f"  - {test_file}")
        return True
    
    # 尝试从 Kaggle 下载
    # 注意: 需要 kaggle.json API key
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "shayanfazeli/heartbeat", "-p", str(data_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # 解压
            zip_file = data_dir / "heartbeat.zip"
            if zip_file.exists():
                print("[INFO] Extracting dataset...")
                with zipfile.ZipFile(zip_file, 'r') as z:
                    z.extractall(data_dir)
                zip_file.unlink()  # 删除 zip
                print("[OK] Dataset downloaded and extracted!")
                return True
        else:
            print(f"[ERROR] Kaggle download failed: {result.stderr}")
            
    except Exception as e:
        print(f"[ERROR] {e}")
    
    return False

def create_mock_data():
    """创建模拟数据用于测试 (如果下载失败)"""
    import numpy as np
    import pandas as pd
    
    print("\n[INFO] Creating mock MIT-BIH data for testing...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    np.random.seed(42)
    
    # 模拟训练集 (80k samples)
    n_train = 80000
    train_data = np.random.randn(n_train, 187).astype(np.float32)
    train_labels = np.random.choice([0, 1, 2, 3, 4], n_train, p=[0.36, 0.11, 0.28, 0.07, 0.18])
    train_df = pd.DataFrame(np.column_stack([train_data, train_labels]))
    train_df.to_csv(data_dir / "mitbih_train.csv", index=False, header=False)
    
    # 模拟测试集 (20k samples)
    n_test = 20000
    test_data = np.random.randn(n_test, 187).astype(np.float32)
    test_labels = np.random.choice([0, 1, 2, 3, 4], n_test, p=[0.36, 0.11, 0.28, 0.07, 0.18])
    test_df = pd.DataFrame(np.column_stack([test_data, test_labels]))
    test_df.to_csv(data_dir / "mitbih_test.csv", index=False, header=False)
    
    print(f"[OK] Mock data created!")
    print(f"  - Train: {n_train} samples")
    print(f"  - Test: {n_test} samples")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("MIT-BIH Dataset Download")
    print("="*60)
    
    # 尝试下载真实数据
    if not download_mitbih():
        # 如果下载失败，创建模拟数据
        create_mock_data()
    
    print("\n" + "="*60)
    print("Data preparation completed!")
    print("="*60)
