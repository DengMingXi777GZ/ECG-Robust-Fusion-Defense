"""
FGSM (Fast Gradient Sign Method) 攻击实现
参考: Goodfellow et al. "Explaining and Harnessing Adversarial Examples" ICLR 2015
"""
import torch
import torch.nn as nn
from attacks.base_attack import BaseAttack


class FGSM(BaseAttack):
    """
    Fast Gradient Sign Method 攻击
    
    公式:
        x_adv = x + ε * sign(∇_x L(f(x), y))
    
    对于目标攻击:
        x_adv = x - ε * sign(∇_x L(f(x), y_target))
    """
    
    def __init__(self, model, device='cpu', eps=0.01):
        super().__init__(model, device, eps)
    
    def generate(self, x, y=None, targeted=False):
        """
        生成 FGSM 对抗样本
        
        Args:
            x: 原始输入, shape [B, C, L]
            y: 标签
            targeted: 是否为目标攻击 (默认False)
        
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
        
        x_temp = x.clone().detach().requires_grad_(True)
        
        # 前向传播
        output = self.model(x_temp)
        loss = nn.CrossEntropyLoss()(output, y)
        
        # 反向传播
        self.model.zero_grad()
        loss.backward()
        
        # 获取梯度符号
        grad_sign = x_temp.grad.data.sign()
        
        # 生成对抗样本
        if targeted:
            # 目标攻击: 向目标类别移动
            x_adv = x - self.eps * grad_sign
        else:
            # 非目标攻击: 远离真实类别
            x_adv = x + self.eps * grad_sign
        
        # 裁剪到有效范围
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
    attacker = FGSM(model, eps=0.01)
    
    # 测试输入
    x = torch.randn(4, 1, 187)
    y = torch.tensor([0, 1, 2, 3])
    
    # 生成对抗样本
    x_adv = attacker.generate(x, y)
    
    print(f"Original shape: {x.shape}")
    print(f"Adversarial shape: {x_adv.shape}")
    print(f"Perturbation max: {(x_adv - x).abs().max().item():.6f}")
    print(f"Perturbation L-inf: {(x_adv - x).abs().max().item():.6f}")
    
    # 验证模型输出变化
    with torch.no_grad():
        out_orig = model(x).argmax(dim=1)
        out_adv = model(x_adv).argmax(dim=1)
    
    print(f"Original predictions: {out_orig}")
    print(f"Adversarial predictions: {out_adv}")
    print(f"Attack success: {(out_orig != out_adv).sum().item()}/{len(y)} samples")
    
    print("\n✅ FGSM test passed!")
