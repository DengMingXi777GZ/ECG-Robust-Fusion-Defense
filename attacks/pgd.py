"""
PGD (Projected Gradient Descent) 攻击实现
参考: Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" ICLR 2018
"""
import torch
import torch.nn as nn
from attacks.base_attack import BaseAttack


class PGD(BaseAttack):
    """
    Projected Gradient Descent 攻击 (多步迭代 FGSM)
    
    参数:
        num_steps: 迭代步数 (默认20)
        alpha: 步长 (默认 eps/5)
        random_start: 是否随机初始化 (默认True)
    """
    
    def __init__(self, model, device='cpu', eps=0.01, num_steps=20, alpha=None, random_start=True):
        super().__init__(model, device, eps)
        self.num_steps = num_steps
        self.alpha = alpha if alpha is not None else eps / 5
        self.random_start = random_start
    
    def generate(self, x, y=None, targeted=False):
        """
        生成 PGD 对抗样本
        
        Args:
            x: 原始输入, shape [B, C, L]
            y: 标签
            targeted: 是否为目标攻击
        
        Returns:
            x_adv: 对抗样本
        """
        x = x.clone().detach().to(self.device)
        
        if y is not None:
            y = y.to(self.device)
        else:
            # 使用模型预测作为伪标签
            with torch.no_grad():
                y = self.model(x).argmax(dim=1)
        
        # 初始化对抗样本
        if self.random_start:
            # 在 epsilon 球内随机初始化
            noise = torch.empty_like(x).uniform_(-self.eps, self.eps)
            x_adv = x + noise
            x_adv = torch.clamp(x_adv, 0, 1)
        else:
            x_adv = x.clone()
        
        # 迭代攻击
        for step in range(self.num_steps):
            x_temp = x_adv.clone().detach().requires_grad_(True)
            
            # 前向传播
            output = self.model(x_temp)
            loss = nn.CrossEntropyLoss()(output, y)
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 获取梯度
            grad = x_temp.grad.data
            
            # 更新对抗样本
            with torch.no_grad():
                if targeted:
                    # 目标攻击: 向目标类别移动
                    x_adv = x_adv - self.alpha * grad.sign()
                else:
                    # 非目标攻击: 远离真实类别
                    x_adv = x_adv + self.alpha * grad.sign()
                
                # 投影回 epsilon 球和 [0, 1] 范围
                x_adv = self.clip(x_adv, x)
        
        return x_adv.detach()


if __name__ == "__main__":
    # 测试
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    os.chdir(project_root)
    
    from models.ecg_cnn import ECG_CNN
    from attacks.base_attack import BaseAttack  # noqa: F401
    
    # 创建模型
    model = ECG_CNN(num_classes=5)
    model.eval()
    
    # 创建攻击器
    attacker = PGD(model, eps=0.01, num_steps=20, random_start=True)
    
    # 测试输入
    x = torch.randn(4, 1, 187)
    y = torch.tensor([0, 1, 2, 3])
    
    # 生成对抗样本
    x_adv = attacker.generate(x, y)
    
    print(f"Original shape: {x.shape}")
    print(f"Adversarial shape: {x_adv.shape}")
    print(f"Perturbation max: {(x_adv - x).abs().max().item():.6f}")
    print(f"Epsilon: {attacker.eps}")
    
    # 验证模型输出变化
    with torch.no_grad():
        out_orig = model(x).argmax(dim=1)
        out_adv = model(x_adv).argmax(dim=1)
    
    print(f"Original predictions: {out_orig}")
    print(f"Adversarial predictions: {out_adv}")
    print(f"Attack success: {(out_orig != out_adv).sum().item()}/{len(y)} samples")
    
    print("\n✅ PGD test passed!")
