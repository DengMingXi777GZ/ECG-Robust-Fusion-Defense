基于**Layer 1已完成**的现状（基线模型93.43%准确率，eps=0.05时PGD ASR=88.58%），以下是**Layer 2：防御层（Defense Layer）**的完整任务清单。

---

# 🛡️ Kimi Code 工作列表文件 #2：防御层 (Defense & Robust Training Layer)

**前置依赖**：必须完成Layer 1（`checkpoints/clean_model.pth` + `data/adversarial/`下的对抗样本）  
**核心目标**：构建能抵抗eps=0.05攻击的鲁棒模型（对抗训练 + NSR正则化）  
**关键设定**：基于Layer 1发现，使用**eps=0.05**作为训练和评估标准（PGD-40 ASR=88.58%，有效且生理可信）

---

## 📥 继承资产清单（来自Layer 1）

**必须存在的文件**（Layer 2启动前检查）：
```bash
checkpoints/
└── clean_model.pth              # 基线模型 (93.43% acc)

data/adversarial/                # Layer 1生成的对抗样本
├── test_fgsm_eps005.pt          # 需重新生成，eps=0.05
├── test_pgd_eps005.pt           # 用于评估
└── test_sap_eps005.pt           # 用于评估

attacks/                         # Layer 1的攻击模块（直接复用）
├── base_attack.py
├── pgd.py                       # 关键：PGD(eps=0.05, steps=40)
└── sap.py                       # 关键：SAP(eps=0.05, steps=40)
```

**Layer 1关键结论**：
- 模型在`eps=0.01`时ASR过低(1.97%)，但在`eps=0.05`时ASR=88.58%（有效攻击）
- 因此Layer 2所有对抗训练使用**eps=0.05**作为扰动预算

---

## 模块6：对抗训练基础设施 (Adversarial Training Core)

### Task 2.1 对抗训练数据集生成器（动态生成）
**文件**：`defense/adv_trainer.py`中的`AdversarialDataset`类  
**继承要求**：使用Layer 1的PGD/SAP实现，但参数改为eps=0.05

```python
class AdversarialDataset(Dataset):
    def __init__(self, clean_dataset, model, attack_method='pgd', 
                 eps=0.05, steps=40, alpha=0.0125):  # alpha=eps/4
        """
        动态生成对抗样本，节省内存
        关键：使用Layer 1验证有效的eps=0.05参数
        """
        self.clean_data = clean_dataset
        self.model = model
        # 直接复用Layer 1的attacks模块
        if attack_method == 'pgd':
            self.attacker = PGD(model, eps=eps, steps=steps, alpha=alpha)
        elif attack_method == 'sap':
            self.attacker = SAP(model, eps=eps, steps=steps)
    
    def __getitem__(self, idx):
        x, y = self.clean_data[idx]
        # 关键：每次getitem生成新的对抗样本（避免过拟合到固定噪声）
        with torch.enable_grad():  # 确保梯度开启
            x_adv = self.attacker.generate(x.unsqueeze(0), 
                                           y.unsqueeze(0)).squeeze(0)
        return x, x_adv, y
```

### Task 2.2 重新生成eps=0.05的对抗样本数据集
**命令**（基于Layer 1的generate脚本，修改eps）：
```bash
python generate_adversarial_dataset.py \
    --checkpoint checkpoints/clean_model.pth \
    --eps 0.05 \
    --pgd-steps 40 \
    --sap-steps 40 \
    --output-dir data/adversarial/eps005/
```

**预期产出**：
```
data/adversarial/eps005/
├── test_fgsm.pt      (21,892条, eps=0.05, ASR~8-30%)
├── test_pgd.pt       (21,892条, eps=0.05, ASR~88%) 
└── test_sap.pt       (21,892条, eps=0.05, ASR~85%)
```

---

## 模块7：对抗训练实现 (Standard AT)

### Task 2.3 标准对抗训练 (Madry's AT)
**文件**：`defense/train_standard_at.py`  
**核心逻辑**：Min-Max优化，混合Clean和Adv样本

```python
def train_epoch(model, loader, optimizer, criterion, eps=0.05):
    model.train()
    total_loss = 0
    
    for x_clean, y in loader:
        x_clean, y = x_clean.cuda(), y.cuda()
        
        # 1. 生成对抗样本（使用当前模型状态）
        attacker = PGD(model, eps=eps, steps=10, alpha=eps/4)  # 训练用10步
        x_adv = attacker.generate(x_clean, y)
        
        # 2. 混合数据（Madry标准做法：各50%）
        x_mixed = torch.cat([x_clean, x_adv], dim=0)
        y_mixed = torch.cat([y, y], dim=0)
        
        # 3. 前向与损失
        output = model(x_mixed)
        loss = criterion(output, y_mixed)
        
        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)
```

**超参数配置**（基于Ma & Liang 2022）：
- Epochs: 50
- Optimizer: Adam, lr=0.001 (前5epoch warmup只用clean数据)
- Epsilon: **0.05**（基于Layer 1结论）
- Attack steps: 10（训练时，比评估时的40步弱，防止过拟合）
- Batch size: 256

**验收标准**：
- Clean Accuracy ≥ 88%（允许比93.43%下降5%以内）
- PGD-20 (eps=0.05) Accuracy ≥ 60%（鲁棒性提升）
- 模型保存：`checkpoints/adv_standard_at.pth`

---

## 模块8：NSR正则化实现 (核心创新)

### Task 2.4 NSR损失计算器
**文件**：`defense/nsr_loss.py`  
**学术来源**：Ma & Liang 2022, Eq.(7)  
**核心公式**：
$$L_{NSR} = (z_y-1)^2 + \sum_{i\neq y}(z_i-0)^2 + \sum_{i\neq y}\max(0,1-z_y+z_i) + \beta \cdot \log(R+1)$$
其中 $R = \frac{\|w_y\|_1 \cdot \epsilon}{|z_y|}$

```python
class NSRLoss(nn.Module):
    def __init__(self, beta=0.4, eps=0.05, num_classes=5):
        super().__init__()
        self.beta = beta      # 正则化系数，MIT-BIH上最佳0.4
        self.eps = eps        # 使用Layer 1验证的0.05
        self.num_classes = num_classes
        self.mse = nn.MSELoss()
    
    def forward(self, model, x, y, output):
        batch_size = y.size(0)
        
        # 1. MSE Loss (One-hot目标)
        y_onehot = F.one_hot(y, self.num_classes).float()
        mse_loss = self.mse(output, y_onehot)
        
        # 2. Margin Loss (仅对正确分类样本)
        z_y = output[range(batch_size), y]
        margins = torch.clamp(1 - z_y.unsqueeze(1) + output, min=0)
        margins[range(batch_size), y] = 0
        margin_loss = margins.sum() / batch_size
        
        # 3. NSR Regularization (关键部分)
        # 计算 ||w_y||_1：对类别y的logit关于输入x的梯度的L1范数
        w_l1 = torch.zeros(batch_size, device=x.device)
        for i in range(batch_size):
            xi = x[i:i+1].clone().detach().requires_grad_(True)
            out = model(xi)
            z = out[0, y[i]]
            grad = torch.autograd.grad(z, xi, create_graph=True)[0]
            w_l1[i] = torch.norm(grad, p=1)
        
        # 计算 R = ||w_y||_1 * eps / |z_y|
        R = (w_l1 * self.eps) / (torch.abs(z_y) + 1e-8)
        nsr_loss = self.beta * torch.mean(torch.log(R + 1))
        
        # 4. 组合（仅对正确分类样本应用NSR和Margin）
        pred = output.argmax(dim=1)
        correct_mask = (pred == y).float().mean()
        
        total_loss = mse_loss + (margin_loss + nsr_loss) * correct_mask
        return total_loss, {
            'mse': mse_loss.item(),
            'margin': margin_loss.item(),
            'nsr': nsr_loss.item()
        }
```

### Task 2.5 NSR训练管道
**文件**：`defense/train_nsr.py`  
**关键差异**：使用NSRLoss替代CrossEntropy，eps=0.05

```python
# 训练配置
criterion = NSRLoss(beta=0.4, eps=0.05)  # 基于Layer 1的eps
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 延迟启动NSR（前10epoch只用MSE，防止初期梯度爆炸）
for epoch in range(50):
    if epoch < 10:
        criterion.beta = 0  # 关闭NSR
    else:
        criterion.beta = 0.4  # 开启NSR
    
    train_epoch(...)
```

**超参数搜索**（必须做）：
- Beta候选值：[0.2, 0.4, 0.6, 0.8, 1.0]
- 评估指标：ACC_robust = sqrt(ACC_clean * AUC_under_attack)
- 选择验证集上ACC_robust最高的beta

---

## 模块9：融合方案（AT + NSR）

### Task 2.6 联合训练 (AT + NSR)
**文件**：`defense/train_at_nsr.py`  
**策略**：对抗训练 + NSR正则化双重防护

```python
def train_combined(model, loader, optimizer, eps=0.05, beta=0.4):
    for x_clean, y in loader:
        # 1. 生成对抗样本（AT部分）
        attacker = PGD(model, eps=eps, steps=10)
        x_adv = attacker.generate(x_clean, y)
        
        # 2. 混合数据
        x_mixed = torch.cat([x_clean, x_adv], dim=0)
        y_mixed = torch.cat([y, y], dim=0)
        
        # 3. 使用NSR Loss（替代标准CE）
        output = model(x_mixed)
        loss, loss_dict = nsr_criterion(model, x_mixed, y_mixed, output)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 模块10：防御评估系统

### Task 2.7 鲁棒性评估框架
**文件**：`evaluation/defense_eval.py`  
**评估标准**：对比Clean模型 vs 防御模型在相同攻击下的表现

```python
def evaluate_robustness(model, test_loader, eps=0.05):
    results = {}
    
    # 测试攻击（使用Layer 1的attacks，eps=0.05）
    attacks = {
        'clean': lambda x, y: (x, y),
        'fgsm': FGSM(model, eps),
        'pgd20': PGD(model, eps, steps=20),    # 评估用20步
        'pgd100': PGD(model, eps, steps=100),  # 强攻击
        'sap': SAP(model, eps, steps=40)
    }
    
    for name, attacker in attacks.items():
        correct = 0
        total = 0
        
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            if name == 'clean':
                x_adv = x
            else:
                x_adv = attacker.generate(x, y)
            
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        results[name] = 100.0 * correct / total
    
    return results
```

**预期结果对比表**（目标）：

| 模型 | Clean | FGSM | PGD-20 | PGD-100 | SAP | ACC_robust |
|------|-------|------|--------|---------|-----|------------|
| **Clean** (Layer 1) | 93.4% | 8.3% | 11.5% | 1.8% | ~10% | ~0.35 |
| **Standard AT** | 88.0% | 75% | 65% | 45% | 60% | ~0.70 |
| **NSR (β=0.4)** | 90.5% | 70% | 72% | 55% | 80% | ~0.75 |
| **AT+NSR** | 87.0% | 78% | 75% | 60% | 82% | ~0.78 |

**ACC_robust计算**（基于Ma论文）：
```python
def compute_acc_robust(clean_acc, adv_accuracies):
    """几何平均：sqrt(clean_acc * mean(adv_accuracies))"""
    import numpy as np
    return np.sqrt(clean_acc * np.mean(adv_accuracies))
```

### Task 2.8 超参数调优脚本
**文件**：`experiments/tune_beta.py`  
**功能**：自动化搜索最佳beta值（NSR）和训练策略

```bash
python experiments/tune_beta.py \
    --betas 0.2 0.4 0.6 0.8 1.0 \
    --eps 0.05 \
    --epochs 50
```

---

## 📤 Layer 2 交付检查清单

完成以下检查后，进入Layer 3（特征融合）：

- [ ] **重新生成eps=0.05对抗样本**：`data/adversarial/eps005/`目录存在3个.pt文件
- [ ] **Standard AT训练完成**：`checkpoints/adv_standard_at.pth`，Clean Acc≥88%，PGD-20≥60%
- [ ] **NSR训练完成**：`checkpoints/nsr_beta0.4.pth`，ACC_robust≥0.75
- [ ] **超参数搜索完成**：`results/beta_tuning_results.json`，确定最佳beta
- [ ] **对比实验完成**：生成对比表格（Clean vs AT vs NSR vs AT+NSR）
- [ ] **鲁棒性曲线**：绘制Accuracy vs Epsilon曲线（eps=0~0.1）

---

## 🔗 与Layer 3的衔接（预览）

Layer 3将使用Layer 2训练好的**鲁棒模型**（如NSR模型）作为特征提取器，结合人工特征（neurokit2提取的RR间期等）构建双分支检测器。因此Layer 2的模型权重是Layer 3的输入。

**关键继承点**：
- Layer 2的`checkpoints/nsr_beta0.4.pth`将作为Layer 3的`deep_branch`预训练权重
- Layer 2的`evaluate_robustness`函数将用于验证融合后的防御效果

---

**预计耗时**：12-16天（5060显卡，含超参数搜索）  
**关键依赖**：必须使用**eps=0.05**（基于Layer 1验证的有效攻击强度）