# 防御层 (Defense Layer) - Layer 2

## 概述

本目录包含 Layer 2 防御层的所有实现，基于 Layer 1 的结论（eps=0.05 是有效且生理可信的攻击强度）。

## 核心模块

### 1. 对抗训练基础设施 (`adv_trainer.py`)

**AdversarialDataset**: 动态生成对抗样本的数据集
- 每次 `__getitem__` 生成新的对抗样本（避免过拟合到固定噪声）
- 直接复用 Layer 1 的 PGD/SAP 实现
- 支持 eps=0.05（基于 Layer 1 验证的有效参数）

**MixedAdversarialDataset**: 混合数据集，用于 Madry's AT

### 2. NSR 损失计算器 (`nsr_loss.py`)

**NSRLoss**: 实现 Ma & Liang 2022 论文中的 NSR 正则化

核心公式:
```
L_NSR = (z_y-1)^2 + Σ_{i≠y}(z_i-0)^2 + Σ_{i≠y}max(0,1-z_y+z_i) + β·log(R+1)
其中 R = ||w_y||_1 · ε / |z_y|
```

参数:
- `beta`: 正则化系数，MIT-BIH 上最佳 0.4
- `eps`: 扰动预算，使用 Layer 1 验证的 0.05
- `num_classes`: 类别数（默认 5）

### 3. 标准对抗训练 (`train_standard_at.py`)

**Madry's AT 实现**

超参数配置:
- Epochs: 50
- Optimizer: Adam, lr=0.001
- Epsilon: 0.05
- Attack steps: 10（训练时）
- Batch size: 256
- Warmup: 前 5 epoch 只用 clean 数据

验收标准:
- Clean Accuracy ≥ 88%
- PGD-20 (eps=0.05) Accuracy ≥ 60%

使用方法:
```bash
python defense/train_standard_at.py \
    --eps 0.05 \
    --epochs 50 \
    --batch-size 256
```

### 4. NSR 训练管道 (`train_nsr.py`)

**NSR 正则化训练**

关键特性:
- 使用 NSRLoss 替代 CrossEntropy
- 延迟启动 NSR（前 10 epoch 只用 MSE，防止初期梯度爆炸）
- 支持超参数搜索（beta: [0.2, 0.4, 0.6, 0.8, 1.0]）

评估指标:
```
ACC_robust = sqrt(ACC_clean * mean(adv_accuracies))
```

使用方法:
```bash
python defense/train_nsr.py \
    --beta 0.4 \
    --eps 0.05 \
    --epochs 50
```

### 5. 联合训练 (`train_at_nsr.py`)

**AT + NSR 融合方案**

策略:
- 对抗训练：生成对抗样本并混合 Clean 和 Adv 数据
- NSR 正则化：使用 NSRLoss 替代标准 CE
- 双重防护：既提高鲁棒性又保持准确率

使用方法:
```bash
python defense/train_at_nsr.py \
    --beta 0.4 \
    --eps 0.05 \
    --epochs 50
```

## 快速开始

### 1. 生成 eps=0.05 的对抗样本

```bash
python generate_adversarial_dataset.py \
    --checkpoint checkpoints/clean_model.pth \
    --eps 0.05 \
    --pgd-steps 40 \
    --sap-steps 40 \
    --output-dir data/adversarial/eps005/
```

### 2. 训练防御模型

**选项 A: 标准对抗训练**
```bash
python defense/train_standard_at.py --epochs 50
```

**选项 B: NSR 训练**
```bash
python defense/train_nsr.py --beta 0.4 --epochs 50
```

**选项 C: 联合训练**
```bash
python defense/train_at_nsr.py --beta 0.4 --epochs 50
```

### 3. 评估鲁棒性

```bash
python evaluation/defense_eval.py --compare
```

### 4. 超参数调优

```bash
python experiments/tune_beta.py \
    --betas 0.2 0.4 0.6 0.8 1.0 \
    --eps 0.05 \
    --epochs 30
```

## 预期结果

| 模型 | Clean | FGSM | PGD-20 | PGD-100 | SAP | ACC_robust |
|------|-------|------|--------|---------|-----|------------|
| **Clean** (Layer 1) | 93.4% | 8.3% | 11.5% | 1.8% | ~10% | ~0.35 |
| **Standard AT** | 88.0% | 75% | 65% | 45% | 60% | ~0.70 |
| **NSR (β=0.4)** | 90.5% | 70% | 72% | 55% | 80% | ~0.75 |
| **AT+NSR** | 87.0% | 78% | 75% | 60% | 82% | ~0.78 |

## 文件结构

```
defense/
├── __init__.py              # 模块初始化
├── README.md                # 本文档
├── adv_trainer.py           # Task 2.1: 对抗训练数据集生成器
├── nsr_loss.py              # Task 2.4: NSR 损失计算器
├── train_standard_at.py     # Task 2.3: 标准对抗训练
├── train_nsr.py             # Task 2.5: NSR 训练管道
└── train_at_nsr.py          # Task 2.6: 联合训练
```

## 参考

- Ma & Liang 2022: "Understanding and Improving the Robustness of Deep Learning"
- Madry et al. 2018: "Towards Deep Learning Models Resistant to Adversarial Attacks"
