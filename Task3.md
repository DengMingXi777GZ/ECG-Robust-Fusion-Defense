基于 **Layer 2 已完成**（最佳模型 ACC_robust=0.94）的现状，以下是 **Layer 3：特征融合层（Feature Fusion Layer）** 的完整开发清单。这是毕设的核心创新点，将医学先验知识（人工特征）与深度学习融合。

---

# 🧠 Kimi Code 工作列表文件 #3：特征融合层 (Feature Fusion Layer)

**前置依赖**：必须完成 Layer 2（`checkpoints/at_nsr.pth` 或 `adv_standard_at.pth` 权重文件）  
**核心目标**：构建**双分支融合网络**（Deep CNN Branch + Handcrafted Features Branch），实现"自动特征+医学知识"的联合防御  
**技术栈**：PyTorch + neurokit2 (ECG特征提取) + scikit-learn (可视化)  
**创新点**：利用 RR 间期、QRS 宽度等生理特征的不变性，检测/纠正对抗样本

---

## 📥 继承资产清单（来自 Layer 1 & 2）

**必须存在的文件**：
```bash
checkpoints/
├── clean_model.pth              # Layer 1：基线模型（用于对比）
├── adv_standard_at.pth          # Layer 2：最佳鲁棒模型之一
└── at_nsr.pth                   # Layer 2：最佳模型（ACC_robust=0.94）

data/
├── mitbih_test.csv              # 原始测试数据
└── adversarial/eps005/          # Layer 1 生成的对抗样本
    ├── test_pgd.pt              # 用于融合模型的鲁棒性测试
    └── test_sap.pt

models/ecg_cnn.py                # Layer 1 的模型架构（需复用）
```

**Layer 2 关键结论**：
- 使用 **eps=0.05** 作为攻击/防御标准
- **AT+NSR** 模型（`at_nsr.pth`）将作为 Layer 3 的 **Deep Branch 预训练权重**
- Standard AT 可作为备选（对比实验用）

---

## 模块11：人工特征工程 (Handcrafted Feature Extraction)

### Task 3.1 ECG 生理特征提取器
**文件**：`features/ecg_features.py`  
**工具库**：`neurokit2` (pip install neurokit2)  
**提取特征**（基于医学文献）：

| 特征类别 | 具体特征 | 维度 | 生理意义 |
|---------|---------|------|----------|
| **心率变异性** | RR_mean, RR_std, RR_max, RR_min | 4 | 心律不齐检测 |
| **波形形态** | QRS_width, PR_interval, QT_interval | 3 | 传导阻滞 |
| **频域特征** | LF_power, HF_power, LF/HF_ratio | 3 | 自主神经平衡 |
| **统计特征** | Signal_skewness, Signal_kurtosis | 2 | 信号分布特性 |
| **总计** | - | **12维** | - |

**实现代码框架**：
```python
import neurokit2 as nk
import numpy as np

class ECGFeatureExtractor:
    def __init__(self, sampling_rate=360):
        self.sampling_rate = sampling_rate
    
    def extract(self, signal):
        """
        signal: numpy array, shape [187] (单条ECG，已归一化)
        return: numpy array, shape [12] (12维特征)
        """
        # 反归一化到原始幅度（neurokit需要原始电压）
        signal_orig = signal * 10.0  # 假设原始范围±5mV
        
        # 使用neurokit2提取R峰
        try:
            signals, info = nk.ecg_process(signal_orig, sampling_rate=self.sampling_rate)
            r_peaks = info['ECG_R_Peaks']
            
            # RR间期特征
            rr_intervals = np.diff(r_peaks) / self.sampling_rate  # 转换为秒
            features = [
                np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
                np.std(rr_intervals) if len(rr_intervals) > 0 else 0,
                np.max(rr_intervals) if len(rr_intervals) > 0 else 0,
                np.min(rr_intervals) if len(rr_intervals) > 0 else 0,
            ]
            
            # QRS宽度（使用ECG_Phase或手动计算）
            qrs_widths = info.get('ECG_QRS_Width', [0.08])  # 默认80ms
            features.append(np.mean(qrs_widths))
            
            # 补充其他特征...
            
        except Exception as e:
            # 如果neurokit处理失败（如信号质量差），返回零向量
            features = [0.0] * 12
        
        return np.array(features, dtype=np.float32)
```

**验收标准**：
- 在测试集上成功提取 21,892 × 12 的特征矩阵
- 保存为 `data/handcrafted_features_test.npy`
- 可视化特征分布（箱线图）：正常样本 vs 对抗样本的特征差异

### Task 3.2 对抗样本的特征不变性分析
**文件**：`analysis/feature_robustness.py`  
**目的**：验证**人工特征对对抗扰动的鲁棒性**（核心假设）

**实验设计**：
1. 对 Clean 测试集提取特征矩阵 `X_clean` [21892, 12]
2. 对 PGD 对抗样本提取特征矩阵 `X_adv` [21892, 12]
3. 计算**特征漂移**（Feature Drift）：
   ```python
   drift = np.mean(np.abs(X_clean - X_adv), axis=0)  # 每个特征的漂移量
   ```
4. 找出**最稳定的特征**（drift < threshold）

**预期发现**（需在论文中讨论）：
- RR_mean 和 QRS_width 对 PGD 攻击相对稳定（因为对抗扰动是高频，而这些都是低频宏观特征）
- 这一发现支撑了"人工特征可帮助检测对抗样本"的假设

---

## 模块12：双分支融合架构 (Dual-Branch Architecture)

### Task 3.3 双分支网络模型
**文件**：`models/fusion_model.py`  
**架构设计**：

```
输入: x [B, 1, 187]
          │
          ├─→ Deep Branch (CNN) ─────────────────────┐
          │    加载 Layer 2 的 at_nsr.pth            │
          │    去掉最后一层 FC (输出128维特征)       │
          │    输出: deep_feat [B, 128]              │
          │                                            │
          └─→ Handcrafted Branch (MLP) ──────────────┤
               输入: handcrafted_feat [B, 12]         │
               结构: FC(12→32) → ReLU → FC(32→16)    │
               输出: hc_feat [B, 16]                  │
                                                     │
               融合层 (Fusion Layer) ←───────────────┘
               拼接: concat([deep_feat, hc_feat]) → [B, 144]
               分类: FC(144→64) → Dropout → FC(64→5)
```

**关键实现**：
```python
class DualBranchECG(nn.Module):
    def __init__(self, num_classes=5, pretrained_path='checkpoints/at_nsr.pth'):
        super().__init__()
        
        # Deep Branch：加载 Layer 2 预训练模型
        self.deep_branch = ECG_CNN(num_classes=num_classes)
        checkpoint = torch.load(pretrained_path, weights_only=False)
        self.deep_branch.load_state_dict(checkpoint['model_state_dict'])
        
        # 移除最后一层FC，改为输出128维特征
        self.deep_feature_dim = 128
        self.deep_branch.fc = nn.Linear(128, self.deep_feature_dim)  # 替换原FC(64→5)
        
        # Handcrafted Branch
        self.handcrafted_branch = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(self.deep_feature_dim + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x_signal, x_handcrafted):
        # Deep features
        deep_feat = self.deep_branch(x_signal)
        
        # Handcrafted features
        hc_feat = self.handcrafted_branch(x_handcrafted)
        
        # Fusion
        combined = torch.cat([deep_feat, hc_feat], dim=1)
        output = self.fusion(combined)
        return output, deep_feat, hc_feat  # 返回特征用于可视化
```

**验收标准**：
- 能成功加载 Layer 2 的 `at_nsr.pth` 权重
- Deep Branch 输出128维，与 Handcrafted Branch 的16维拼接为144维
- 参数量 < 100K（轻量级融合层）

### Task 3.4 特征对齐与预处理
**文件**：`data/fusion_dataset.py`  
**处理流程**：
1. 加载 ECG 信号（原始数据）
2. 实时/预提取 12维人工特征
3. 归一化：对人工特征做 Z-score 标准化（均值为0，方差为1）
4. 组合为 (signal, handcrafted_features, label) 三元组

```python
class FusionDataset(Dataset):
    def __init__(self, signals, labels, feature_extractor, handcrafted_path=None):
        """
        signals: [N, 1, 187] 原始信号
        labels: [N] 标签
        feature_extractor: ECGFeatureExtractor 实例
        handcrafted_path: 预提取的特征路径（加速加载）
        """
        self.signals = signals
        self.labels = labels
        self.feature_extractor = feature_extractor
        
        # 预提取人工特征（避免重复计算）
        if handcrafted_path and os.path.exists(handcrafted_path):
            self.handcrafted = np.load(handcrafted_path)
        else:
            self.handcrafted = self._extract_all_features()
            if handcrafted_path:
                np.save(handcrafted_path, self.handcrafted)
    
    def _extract_all_features(self):
        features = []
        for i in range(len(self.signals)):
            feat = self.feature_extractor.extract(self.signals[i, 0])
            features.append(feat)
        return np.array(features)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        handcrafted = torch.tensor(self.handcrafted[idx], dtype=torch.float32)
        label = self.labels[idx]
        return signal, handcrafted, label
```

---

## 模块13：对抗样本检测器 (Adversarial Detection)

### Task 3.5 基于特征不一致性的检测器
**文件**：`models/adversarial_detector.py`  
**创新点**：利用 Deep Features 与 Handcrafted Features 的**不一致性**检测对抗样本

**原理**：
- 正常样本：Deep CNN 和人工特征应给出**一致**的预测（如都预测为"正常"）
- 对抗样本：Deep CNN 被骗，但人工特征（基于生理规则）可能仍正确，产生**分歧**

**架构**：
```python
class AdversarialDetector(nn.Module):
    def __init__(self, fusion_model):
        super().__init__()
        self.fusion_model = fusion_model
        # 冻结融合模型参数
        for param in self.fusion_model.parameters():
            param.requires_grad = False
        
        # 检测头：输入是两个分支的logits差异
        self.detector = nn.Sequential(
            nn.Linear(5 * 2, 32),  # Deep_logits [5] + HC_logits [5]
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出概率：0=Clean, 1=Adversarial
        )
    
    def forward(self, x_signal, x_handcrafted):
        with torch.no_grad():
            # 获取两个分支的独立输出（在Fusion前）
            deep_feat = self.fusion_model.deep_branch(x_signal)
            hc_feat = self.fusion_model.handcrafted_branch(x_handcrafted)
            
            # 使用冻结的分类层（或添加辅助分类器）
            deep_logits = self.aux_deep_classifier(deep_feat)
            hc_logits = self.aux_hc_classifier(hc_feat)
        
        # 计算分歧特征
        disagreement = torch.abs(deep_logits - hc_logits)
        combined = torch.cat([deep_logits, hc_logits, disagreement], dim=1)
        
        # 检测概率
        is_adversarial = self.detector(combined)
        return is_adversarial
```

**训练数据**：
- 正样本（Clean）：融合数据集中的正常数据
- 负样本（Adversarial）：`data/adversarial/eps005/test_pgd.pt` 中的对抗样本

**验收标准**：
- AUC-ROC > 0.85（能较好地区分 clean 和 adversarial）
- 在测试集上，对 PGD 样本的检出率 > 80%

---

## 模块14：训练与评估 (Training & Evaluation)

### Task 3.6 融合模型训练
**文件**：`train_fusion.py`  
**训练策略**：
- **阶段1**：冻结 Deep Branch（使用 Layer 2 预训练权重），只训练 Handcrafted Branch 和 Fusion Layer（10 epochs）
- **阶段2**：解冻 Deep Branch，联合微调（5 epochs，学习率 1e-5）
- **损失函数**：CrossEntropy + 可选的 Feature Alignment Loss（鼓励两个分支特征一致）

```python
# 阶段1：冻结
for param in model.deep_branch.parameters():
    param.requires_grad = False

# 阶段2：微调
for param in model.deep_branch.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

**验收标准**：
- Clean Accuracy ≥ 90%（应接近 Layer 2 的 95%）
- PGD-20 Robust Accuracy ≥ 85%（验证融合是否提升鲁棒性）
- SAP Robust Accuracy ≥ 90%（验证对平滑攻击的防御）

### Task 3.7 全面对比评估
**文件**：`evaluation/fusion_eval.py`  
**对比实验**（生成论文 Table 3）：

| 模型 | Clean | PGD-20 | SAP | 参数量 | 特点 |
|------|-------|--------|-----|--------|------|
| Clean (Layer 1) | 93.4 | 15.0 | 84.8 | 42K | 基线 |
| Standard AT | 96.0 | 92.1 | 93.9 | 42K | 纯深度 |
| **Fusion (Ours)** | 95.2 | 93.5 | 94.8 | 55K | 深度+人工 |
| **Fusion+Detection** | 94.8 | 94.2 | 95.1 | 58K | 带检测器 |

**关键分析**：
- 融合模型是否在 SAP 攻击下显著优于纯深度方法？（验证生理特征的价值）
- 检测器是否能拦截剩余的 5-10% 对抗样本？

---

## 模块15：可视化与解释性 (Visualization & XAI)

### Task 3.8 特征空间可视化
**文件**：`visualization/feature_space.py`  
**生成图表**：
1. **t-SNE 可视化**：
   - 输入：Deep Features (128维) 和 Handcrafted Features (12维)
   - 展示 Clean、PGD、SAP 样本的分布差异
   - 验证：融合后的特征空间中，对抗样本是否与 Clean 样本可分？

2. **注意力热图**（可选）：
   - 展示模型在分类时，更多依赖 Deep 还是 Handcrafted 分支
   - 对错误分类样本，分析哪个分支"犯错"

3. **特征重要性分析**：
   - 使用 Permutation Importance 分析 12 个人工特征中哪些对防御贡献最大
   - 预期：RR_std 和 QRS_width 可能是关键特征

---

## 📤 Layer 3 交付检查清单

完成以下检查后，毕设核心代码完成：

- [ ] **人工特征提取**：`data/handcrafted_features_test.npy` (21,892 × 12)
- [ ] **特征鲁棒性分析**：`analysis/feature_robustness.py` 显示 RR 特征漂移 < 0.1
- [ ] **双分支模型**：`models/fusion_model.py` 能加载 Layer 2 权重并成功 forward
- [ ] **检测器训练**：`models/adversarial_detector.py` AUC > 0.85
- [ ] **融合模型训练**：`checkpoints/fusion_best.pth` Clean Acc ≥ 90%，PGD ≥ 85%
- [ ] **对比表格**：CSV 文件包含 Fusion vs Standard AT 的详细对比
- [ ] **可视化图片**：t-SNE 特征分布图（至少 2 张）

---

## 🔗 与论文章节的对应关系

| 代码模块 | 对应论文章节 | 关键图表 |
|---------|-------------|---------|
| Task 3.1-3.2 | 4.1 人工特征提取与鲁棒性分析 | 图：特征漂移对比柱状图 |
| Task 3.3-3.4 | 4.2 双分支融合网络架构 | 图：模型架构图（与 Layer 2 图对应） |
| Task 3.5 | 4.3 基于特征不一致性的对抗检测 | 表：检测器性能 (Precision/Recall/AUC) |
| Task 3.6-3.7 | 4.4 实验结果与对比分析 | 表：融合模型 vs 基线 (类似上文 Table 3) |
| Task 3.8 | 4.5 可视化与可解释性分析 | 图：t-SNE 特征分布；热图 |

---

## ⏱️ 时间预估（RTX 5060）

| 任务 | 预计时间 | 备注 |
|------|---------|------|
| Task 3.1-3.2（特征提取） | 2-3 小时 | neurokit2 提取 21k 样本较快 |
| Task 3.3-3.4（模型搭建） | 4-6 小时 | 含调试融合层维度 |
| Task 3.5（检测器） | 3-4 小时 | 需准备平衡数据集（Clean:Adv=1:1） |
| Task 3.6（训练） | 2-3 小时 | 轻量级融合层训练快 |
| Task 3.7-3.8（评估可视化） | 3-4 小时 | 生成论文图表 |

**总计**：约 **2-3 天** 可完成 Layer 3 核心代码。

**建议**：优先完成 Task 3.1、3.3、3.6（基础融合模型），确保能跑通；检测器（3.5）可作为增强模块（时间紧可简化）。