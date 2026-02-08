我将为你生成**三个独立的开发清单（Layer 1/2/3）**，每个对应项目的一层架构。这是**第一个清单：攻击层（Layer 1）**，

---

# 🎯 Kimi Code 工作列表文件 #1：攻击层 (Adversarial Attacks Layer)

**目标**：构建能生成生理可信对抗样本的攻击系统（FGSM基础 + SAP核心）  
**技术栈**：PyTorch 2.0+ | torchattacks (参考) | neurokit2 | matplotlib  
**交付物**：`attacks/` 模块 + 可视化报告 + 对抗样本数据集  

---

## 模块1：数据基础设施 (Data Pipeline)

### Task 1.1 MIT-BIH 数据加载器
**文件**：`data/mitbih_loader.py`  
**输入**：Kaggle下载的 `mitbih_train.csv` / `mitbih_test.csv`（包含187采样点+标签）  
**输出**：`torch.utils.data.Dataset` 子类  

**验收标准**：
```python
# 必须支持的API
train_set = MITBIHDataset(csv_path='mitbih_train.csv', transform='normalize')
loader = DataLoader(train_set, batch_size=32)
x, y = next(iter(loader))  # x.shape = [32, 1, 187], y.shape = [32]
```

**关键代码要求**：
- 归一化：Min-Max到 `[0, 1]`（与Han论文一致）
- 标签映射：MIT-BIH的5类（N,S,V,F,Q）转为0-4整数
- 预加载：支持`preload=True`将全数据载入内存（MIT-BIH很小，只有10万条）

---

## 模块2：基线分类模型 (Victim Model)

### Task 1.2 1D-CNN 基线模型
**文件**：`models/ecg_cnn.py`  
**架构**（与Ma & Liang 2022一致以便对比）：
```python
class ECG_CNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 4 Conv Blocks: Conv1d -> BN -> ReLU -> MaxPool
        # Block 1: in=1, out=16, kernel=7, stride=1, padding=3
        # Block 2: in=16, out=32, kernel=5, stride=1, padding=2  
        # Block 3: in=32, out=64, kernel=3, stride=1, padding=1
        # Block 4: in=64, out=128, kernel=3, stride=1, padding=1
        # GlobalAvgPool -> FC(128->64) -> FC(64->5)
```

**验收标准**：
- 参数量：< 500K（5060显卡友好）
- Clean Accuracy：在测试集上 **≥ 91%**（MIT-BIH baseline要求）
- 保存路径：`checkpoints/clean_model.pth`

---

## 模块3：攻击算法实现 (Core Algorithms)

### Task 1.3 基础攻击基类
**文件**：`attacks/base_attack.py`  
**抽象接口**：
```python
class BaseAttack(ABC):
    def __init__(self, model, device, eps=0.01):
        self.model = model.eval()
        self.device = device
        self.eps = eps
    
    @abstractmethod
    def generate(self, x, y=None, targeted=False):
        """返回对抗样本 x_adv，与x同shape"""
        pass
    
    def clip(self, x_adv, x_orig):
        """投影回epsilon球和[0,1]范围"""
        return torch.clamp(x_adv, x_orig-self.eps, x_orig+self.eps).clamp(0,1)
```

### Task 1.4 FGSM 实现（热身验证）
**文件**：`attacks/fgsm.py`  
**公式**：`x_adv = x + ε * sign(∇_x L(f(x), y))`  
**特殊要求**：
- 支持`targeted`模式（若targeted=True，则梯度减而非加）
- 单步完成，无迭代

**验收测试**：
```python
# 在测试集上运行
attacker = FGSM(model, eps=0.01)
x_adv = attacker.generate(x, y)
# 验证：模型准确率应从91%降至<20%
```

### Task 1.5 PGD 实现（标准白盒攻击）
**文件**：`attacks/pgd.py`  
**参数**：
- `num_steps`: 迭代步数（默认20，评估时用100）
- `alpha`: 步长（默认0.002，即eps/5）
- `random_start`: True（随机初始化在epsilon球内）

**算法流程**：
```python
x_adv = x + random_noise(-eps, eps)
for t in range(num_steps):
    grad = compute_gradient(loss(f(x_adv), y), x_adv)
    x_adv = x_adv + alpha * sign(grad)
    x_adv = clip(x_adv, x, eps)  # 投影回Linf球
```

### Task 1.6 SAP 平滑攻击（核心创新复现）
**文件**：`attacks/sap.py`  
**论文来源**：Han et al. Nature Medicine 2020  
**关键区别**：传统PGD优化`x_adv`，SAP优化**平滑扰动参数θ**，然后卷积

**实现步骤**：

1. **多尺度高斯核定义**（在`__init__`中预计算）：
```python
self.kernel_sizes = [5, 7, 11, 15, 19]
self.sigmas = [1.0, 3.0, 5.0, 7.0, 10.0]
self.kernels = [self._gaussian_kernel(s, sig) for s, sig in zip(sizes, sigmas)]
```

2. **前向生成函数**：
```python
def generate(self, x, y, num_steps=40):
    # 1. 初始化theta（可学习扰动）
    theta = torch.zeros_like(x, requires_grad=True)
    
    # 2. 可选：用PGD初始化theta（加速收敛）
    with torch.no_grad():
        init_perturb = self._pgd_init(x, y)  # 快速10步PGD
        theta.data = init_perturb
    
    optimizer = torch.optim.Adam([theta], lr=0.01)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # 3. 应用平滑：x_adv = x + mean(conv(theta, kernel_i))
        perturb_smooth = torch.zeros_like(x)
        for k in self.kernels:
            perturb_smooth += F.conv1d(theta, k.to(x.device), padding='same')
        perturb_smooth /= len(self.kernels)
        
        x_adv = x + perturb_smooth
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # 4. 计算损失并反向传播到theta
        loss = -F.cross_entropy(self.model(x_adv), y)  # 最大化损失
        loss.backward()
        optimizer.step()
        
        # 5. 约束theta在eps范围内（Linf约束）
        with torch.no_grad():
            theta.data = torch.clamp(theta.data, -self.eps, self.eps)
    
    return x + self._apply_smoothing(theta).detach()
```

3. **平滑度评估函数**（内部验证用）：
```python
def smoothness_metric(delta):
    """delta: 扰动信号 [1, 1, 187]"""
    diff = delta[0, 0, 1:] - delta[0, 0, :-1]
    return torch.var(diff).item()  # 目标：<0.001（非常平滑）
```

**验收标准**：
- 对抗成功率（ASR）> 80%
- 平滑度（variance of diff）< PGD攻击的10%
- 人类肉眼无法区分（通过后续可视化验证）

---

## 模块4：攻击评估系统 (Evaluation)

### Task 1.7 攻击指标计算器
**文件**：`evaluation/attack_metrics.py`  
**必须实现的指标**：

| 指标 | 函数名 | 计算公式 | 目标值 |
|------|--------|----------|--------|
| ASR | `attack_success_rate()` | 被误分类的对抗样本比例 | >80% |
| L2扰动 | `perturbation_l2()` | `||x_adv - x||_2 / sqrt(dim)` | <0.05 |
| Linf扰动 | `perturbation_linf()` | `max(abs(x_adv - x))` | <=eps (0.01) |
| SNR | `signal_noise_ratio()` | `20*log10(std(x)/std(delta))` | >20dB |
| 平滑度 | `smoothness()` | `var(diff(delta))` | <1e-4 |

### Task 1.8 可视化工具
**文件**：`evaluation/visualizer.py`  
**必须生成的图表**：

1. **波形对比图**（参考Han论文Fig.1）：
   - 子图1：原始ECG（蓝色）+ 对抗ECG（红色半透明叠加）
   - 子图2：扰动波形（单独显示，验证平滑性）
   
2. **攻击强度曲线**：
   - X轴：epsilon (0~0.05)
   - Y轴：模型准确率
   - 对比曲线：Clean vs FGSM vs PGD vs SAP

3. **频谱分析**（验证SAP的平滑性）：
   - 使用FFT对比PGD和SAP扰动的频谱（SAP高频成分应更少）

**输出格式**：`results/figures/attack_*.png`，300dpi，适合论文插入

---

## 模块5：集成与数据生成

### Task 1.9 对抗样本数据集生成器
**文件**：`generate_adversarial_dataset.py`  
**功能**：
- 加载训练好的Clean模型
- 对测试集生成三种攻击样本：
  - `test_pgd.pt` (eps=0.01, 20-steps)
  - `test_sap.pt` (eps=0.01, 40-steps, 多尺度高斯)
  - `test_fgsm.pt` (eps=0.01)
- 保存格式：`torch.save({'x_adv': tensor, 'y': tensor, 'x_orig': tensor}, file)`

**用途**：这些`.pt`文件将直接用于**Layer 2（防御层）**的对抗训练

---

## 🏁 Layer 1 交付检查清单

完成以下检查项后，Layer 1结束，可进入Layer 2：

- [ ] `python train_baseline.py` 运行后测试集acc ≥ 91%
- [ ] `python attacks/sap.py --test` 能生成单条对抗样本并显示平滑度<1e-4
- [ ] `python evaluate_attacks.py` 输出ASR表格（PGD vs SAP对比）
- [ ] `results/` 目录包含至少3张可视化图片（波形对比、准确率曲线、扰动平滑度）
- [ ] `data/adversarial/` 目录包含生成的`.pt`对抗样本文件（供后续防御训练使用）

---

请回复**"继续"**，我将提供**Layer 2：防御层**（对抗训练 + NSR正则化实现）的详细工作列表。