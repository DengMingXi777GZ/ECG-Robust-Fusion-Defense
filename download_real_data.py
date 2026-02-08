"""
使用 KaggleHub 下载 MIT-BIH 真实数据集
"""
import kagglehub
import shutil
from pathlib import Path
import pandas as pd

print("="*60)
print("Downloading MIT-BIH Arrhythmia Dataset from Kaggle")
print("="*60)

# 下载数据集
print("\n[1/3] Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("shayanfazeli/heartbeat")
print(f"Downloaded to: {path}")

# 检查下载的文件
import os
print("\n[2/3] Checking downloaded files...")
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"  - {file}: {size:.2f} MB")

# 移动到 data 目录
print("\n[3/3] Moving files to data/ directory...")
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

csv_files = list(Path(path).glob("*.csv"))
for csv_file in csv_files:
    dest = data_dir / csv_file.name
    shutil.copy2(csv_file, dest)
    print(f"  Copied: {csv_file.name} -> data/")

# 验证
print("\n" + "="*60)
print("Download Complete!")
print("="*60)

for csv_file in sorted(data_dir.glob("mitbih*.csv")):
    df = pd.read_csv(csv_file, header=None)
    print(f"\n{csv_file.name}:")
    print(f"  Shape: {df.shape}")
    print(f"  Samples: {len(df)}")
    print(f"  Features: {df.shape[1]-1} (+ 1 label)")
