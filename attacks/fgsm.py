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

    """
    Fast Gradient Sign Method (FGSM) 攻击原理：
    
    1. 线性性质假设：FGSM 基于对抗样本研究中的一个核心观察——神经网络在高维空间中往往表现出
       很强的线性特征。这意味着即使是很小的扰动，只要方向正确，也能在输出端产生巨大的变化。
    
    2. 梯度方向：在模型训练中，我们沿梯度负方向更新权重以减小损失。在攻击中，我们沿
       梯度正方向 (sign(∇_x L)) 扰动输入 x。这被称为“梯度上升”，目的是最大化模型的分类损失。
    
    3. L-infinity 范数约束：为了确保生成的扰动肉眼难辨（或在 ECG 数据中保持波形基本特征），
       FGSM 限制扰动的 L-infinity 范数不超过 ε。即输入向量的每一维改变量的绝对值均相等且等于 ε。
    
    4. 计算效率：FGSM 是“单步”攻击，仅需一次前向传播和一次反向传播，不需要迭代，计算速度极快。

    公式：
        非目标攻击：x_adv = x + ε * sign(∇_x L(f(x), y))
        目标攻击：x_adv = x - ε * sign(∇_x L(f(x), y_target))
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

        '''.clone() (克隆)：

        在内存中开辟新空间，复制 x 的内容。
        这可以防止后续对 x 的修改影响到原始输入数据（因为 Python 中是通过引用传递对象的）。
        .detach() (分离)：

        将该张量从当前的计算图中分离出来。
        它会截断梯度追踪，确保这个副本不再参与之前的梯度计算，避免在生成对抗样本时出现内存泄漏或计算冲突。
        .to(self.device) (移动模型)：

        将张量移动到指定的计算设备（如 CPU 或 GPU/CUDA）。
        在 PyTorch 中，输入数据必须与模型权重位于同一设备上才能进行计算。`'''
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
    '''
    标签的意义不同：

    在非目标攻击中，y 是正确的标签。我们用 + 号（梯度上升）是为了增加损失，让模型远离正确的预测。
    在目标攻击中，传入的 y 是攻击者指定的错误标签（例如：把“心脏病”波形改为“正常”波形）。
    最小化目标损失：

    当我们执行 x_adv = x - self.eps * grad_sign 时，我们是在对输入 x 进行梯度下降。
    这会减小模型输出与目标（错误）标签之间的损失（Loss）。
    结果：模型会以极高的置信度认为这个样本属于那个错误的类型。
    '''
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
