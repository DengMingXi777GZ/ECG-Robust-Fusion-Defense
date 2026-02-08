# ECG 对抗攻击系统 - Layer 1: 攻击层

## 项目概述

本项目实现了针对心电图(ECG)分类模型的对抗攻击系统，包括基础攻击(FGSM、PGD)和核心创新方法 SAP(Smooth Adversarial Perturbation)。

参考论文:
- Han et al. "Deep learning models for ECGs are susceptible to adversarial attacks" Nature Medicine 2020 (SAP)
- Ma & Liang 2022 (1D-CNN 架构)

## 项目结构

```
.
├── data/                       # 数据模块
│   ├── mitbih_loader.py       # MIT-BIH 数据集加载器
│   └── adversarial/           # 对抗样本数据集输出目录
├── models/                     # 模型模块
│   └── ecg_cnn.py             # 1D-CNN 基线模型
├── attacks/                    # 攻击算法模块
│   ├── __init__.py
│   ├── base_attack.py         # 攻击基类
│   ├── fgsm.py                # FGSM 攻击
│   ├── pgd.py                 # PGD 攻击
│   └── sap.py                 # SAP 平滑攻击
├── evaluation/                 # 评估模块
│   ├── attack_metrics.py      # 攻击指标计算
│   └── visualizer.py          # 可视化工具
├── checkpoints/               # 模型检查点保存目录
├── results/                   # 结果输出目录
│   └── figures/              # 可视化图片
├── train_baseline.py          # 基线模型训练脚本
├── evaluate_attacks.py        # 攻击效果评估脚本
├── generate_adversarial_dataset.py  # 对抗样本数据集生成器
└── test_attacks.py           # 攻击算法测试脚本
```

## 已实现的模块

### 1. 数据基础设施 (data/mitbih_loader.py)

- **MITBIHDataset**: MIT-BIH 心律失常数据集加载器
  - 支持 Min-Max 归一化到 [0, 1]
  - 标签映射: 5 类 (N,S,V,F,Q) -> 0-4
  - 支持预加载模式 (preload=True)
  - 输出形状: [B, 1, 187]

### 2. 基线分类模型 (models/ecg_cnn.py)

- **ECG_CNN**: 1D-CNN 分类模型
  - 架构: 4 Conv Blocks -> GlobalAvgPool -> 2 FC layers
  - 参数量: ~42K (< 500K)
  - 目标准确率: >= 91% (MIT-BIH)

### 3. 攻击算法 (attacks/)

#### BaseAttack (base_attack.py)
- 抽象基类，定义标准接口
- 提供 `clip()` 方法和梯度计算功能

#### FGSM (fgsm.py)
- 快速梯度符号方法
- 公式: `x_adv = x + ε * sign(∇_x L(f(x), y))`
- 支持目标攻击和非目标攻击

#### PGD (pgd.py)
- 投影梯度下降 (多步 FGSM)
- 参数: num_steps=20, alpha=ε/5
- 支持随机初始化 (random_start=True)

#### SAP (sap.py) - 核心创新
- 平滑对抗扰动攻击
- 优化平滑扰动参数 θ，而非直接优化 x_adv
- 多尺度高斯核平滑: kernel_sizes=[5,7,11,15,19], sigmas=[1,3,5,7,10]
- 平滑度显著优于 PGD (通常 < 10%)

### 4. 评估系统 (evaluation/)

#### 攻击指标 (attack_metrics.py)
- **ASR**: 攻击成功率 (%)
- **L2**: L2 扰动 (归一化)
- **Linf**: L-infinity 扰动
- **SNR**: 信噪比 (dB)
- **Smoothness**: 扰动平滑度 (差分方差)

#### 可视化工具 (visualizer.py)
- **波形对比图**: 原始 ECG + 对抗 ECG + 扰动
- **攻击强度曲线**: 准确率 vs epsilon
- **频谱分析**: FFT 对比 PGD vs SAP
- **指标表格**: 多攻击方法对比

### 5. 集成脚本

- **train_baseline.py**: 训练基线模型
- **evaluate_attacks.py**: 攻击效果评估
- **generate_adversarial_dataset.py**: 生成对抗样本数据集 (.pt 格式)

## 使用方法

### 1. 环境准备

确保已安装 PyTorch 和相关依赖:
```bash
conda activate deepl
```

### 2. 训练基线模型

```bash
python train_baseline.py --epochs 30 --batch-size 128
```

检查点将保存到 `checkpoints/clean_model.pth`

### 3. 测试攻击算法

```bash
python test_attacks.py
```

### 4. 评估攻击效果

```bash
python evaluate_attacks.py --checkpoint checkpoints/clean_model.pth --visualize
```

结果将保存到 `results/figures/`:
- `attack_waveform_sap.png`: SAP 波形对比
- `attack_waveform_pgd.png`: PGD 波形对比
- `attack_frequency_analysis.png`: 频谱分析
- `attack_metrics_table.png`: 指标对比表

### 5. 生成对抗样本数据集

```bash
python generate_adversarial_dataset.py --checkpoint checkpoints/clean_model.pth
```

输出文件将保存到 `data/adversarial/`:
- `test_fgsm.pt`
- `test_pgd.pt`
- `test_sap.pt`

## 验收标准检查清单

- [x] `python train_baseline.py` 运行后测试集 acc ≥ 91%
- [x] `python attacks/sap.py --test` 能生成单条对抗样本并显示平滑度 < 1e-4
- [x] `python evaluate_attacks.py` 输出 ASR 表格（PGD vs SAP 对比）
- [x] `results/` 目录包含至少 3 张可视化图片（波形对比、准确率曲线、扰动平滑度）
- [x] `data/adversarial/` 目录包含生成的 `.pt` 对抗样本文件

## Layer 2 准备工作

生成的对抗样本数据集 (`data/adversarial/*.pt`) 将直接用于 Layer 2 的对抗训练:
- 格式: `torch.save({'x_adv': tensor, 'y': tensor, 'x_orig': tensor}, file)`
- 包含 FGSM、PGD、SAP 三种攻击样本

## 测试结果

```
FGSM Test: perturbation_max = 0.010000
PGD Test: perturbation_max = 0.010000
SAP Test: 
  - perturbation_max = 0.010000
  - smoothness = 0.000018 (SAP) vs 0.000121 (PGD)
  - smoothness_ratio = 0.15 (SAP is 6.7x smoother than PGD)
```

SAP 攻击显著优于 PGD 在平滑度方面，同时保持相同的 L-infinity 约束。
