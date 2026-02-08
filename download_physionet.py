"""
从 PhysioNet 下载 MIT-BIH 心律失常数据集并预处理
"""
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

print("="*60)
print("Downloading MIT-BIH from PhysioNet")
print("="*60)

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# MIT-BIH 记录列表
# 训练集: 前 200 个记录 (除了 102, 104, 107, 217 是起搏器)
# 测试集: 后 50 个记录
train_records = [
    100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214,
    215, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234
]

test_records = [
    100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214,
    215, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234
]

def download_record(record_id, db='mitdb'):
    """下载单个记录"""
    try:
        record = wfdb.rdrecord(str(record_id), pn_dir=f'{db}/{record_id}.z')
        annotation = wfdb.rdann(str(record_id), 'atr', pn_dir=f'{db}/{record_id}.z')
        return record, annotation
    except Exception as e:
        print(f"Error downloading {record_id}: {e}")
        return None, None

def extract_beats(record, annotation, window=187):
    """提取心跳片段"""
    signals = record.p_signal[:, 0]  # 使用第一通道 (MLII)
    samples = annotation.sample
    symbols = annotation.symbol
    
    beats = []
    labels = []
    
    for i, (pos, symbol) in enumerate(zip(samples, symbols)):
        # 只使用正常和异常心跳
        if symbol not in ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', 'P', 'f', 'Q']:
            continue
        
        # 提取窗口
        start = pos - window // 2
        end = pos + window // 2
        
        if start < 0 or end > len(signals):
            continue
        
        beat = signals[start:end]
        
        # 归一化
        beat_min, beat_max = beat.min(), beat.max()
        if beat_max > beat_min:
            beat = (beat - beat_min) / (beat_max - beat_min)
        
        beats.append(beat)
        
        # 标签映射
        if symbol in ['N', 'L', 'R', 'e', 'j']:  # 正常
            labels.append(0)
        elif symbol in ['A', 'a', 'J', 'S']:  # 房性早搏
            labels.append(1)
        elif symbol in ['V', 'E']:  # 室性早搏
            labels.append(2)
        elif symbol in ['F']:  # 融合心跳
            labels.append(3)
        else:  # 其他
            labels.append(4)
    
    return np.array(beats), np.array(labels)

# 下载并处理数据
print("\n[1/2] Processing training data...")
train_beats = []
train_labels = []

for rec_id in tqdm(train_records[:20]):  # 前20个作为训练集
    record, annotation = download_record(rec_id)
    if record is not None:
        beats, labels = extract_beats(record, annotation)
        if len(beats) > 0:
            train_beats.append(beats)
            train_labels.append(labels)

if train_beats:
    train_data = np.vstack(train_beats)
    train_labels = np.hstack(train_labels)
    train_df = pd.DataFrame(np.column_stack([train_data, train_labels]))
    train_df.to_csv(data_dir / "mitbih_train.csv", index=False, header=False)
    print(f"  Saved: mitbih_train.csv ({len(train_df)} samples)")

print("\n[2/2] Processing test data...")
test_beats = []
test_labels = []

for rec_id in tqdm(test_records[:10]):  # 前10个作为测试集
    record, annotation = download_record(rec_id)
    if record is not None:
        beats, labels = extract_beats(record, annotation)
        if len(beats) > 0:
            test_beats.append(beats)
            test_labels.append(labels)

if test_beats:
    test_data = np.vstack(test_beats)
    test_labels = np.hstack(test_labels)
    test_df = pd.DataFrame(np.column_stack([test_data, test_labels]))
    test_df.to_csv(data_dir / "mitbih_test.csv", index=False, header=False)
    print(f"  Saved: mitbih_test.csv ({len(test_df)} samples)")

print("\n" + "="*60)
print("Done!")
print("="*60)
