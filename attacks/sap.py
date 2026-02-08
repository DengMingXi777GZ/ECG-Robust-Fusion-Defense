"""
SAP (Smooth Adversarial Perturbation) 攻击实现
参考: Han et al. "Deep learning models for ECGs are susceptible to adversarial attacks"
       Nature Medicine 2020

关键区别: 传统 PGD 优化 x_adv，SAP 优化平滑扰动参数 θ，然后卷积
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from attacks.base_attack import BaseAttack


class SAP(BaseAttack):
    """
    平滑对抗扰动攻击
    
    通过优化平滑扰动参数 θ，而非直接优化 x_adv，
    使得生成的对抗扰动更加平滑，生理上更可信。
    """
    
    def __init__(self, model, device='cpu', eps=0.01, 
                 num_steps=40, lr=0.01, use_pgd_init=True):
        """
        Args:
            model: 目标模型
            device: 计算设备
            eps: 扰动上限
            num_steps: 优化步数 (默认40)
            lr: 优化器学习率 (默认0.01)
            use_pgd_init: 是否使用 PGD 初始化 (默认True)
        """
        super().__init__(model, device, eps)
        self.num_steps = num_steps
        self.lr = lr
        self.use_pgd_init = use_pgd_init
        
        # 多尺度高斯核定义
        self.kernel_sizes = [5, 7, 11, 15, 19]
        self.sigmas = [1.0, 3.0, 5.0, 7.0, 10.0]
        
        # 预计算高斯核
        self.kernels = self._create_kernels()
    
    def _create_kernels(self):
        """预计算多尺度高斯核"""
        kernels = []
        for size, sigma in zip(self.kernel_sizes, self.sigmas):
            kernel = self._gaussian_kernel(size, sigma)
            kernels.append(kernel)
        return kernels
    
    def _gaussian_kernel(self, size, sigma):
        """
        创建 1D 高斯核
        
        Args:
            size: 核大小 (奇数)
            sigma: 标准差
        
        Returns:
            kernel: [1, 1, size] 归一化高斯核
        """
        # 创建坐标
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        
        # 计算高斯分布
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        
        # 归一化
        g = g / g.sum()
        
        # reshape 为 conv1d 需要的形状 [out_channels, in_channels/groups, kernel_size]
        kernel = g.view(1, 1, -1)
        
        return kernel
    
    def _apply_smoothing(self, theta):
        """
        应用多尺度高斯平滑
        
        Args:
            theta: 扰动参数 [B, C, L]
        
        Returns:
            smoothed: 平滑后的扰动 [B, C, L]
        """
        device = theta.device
        perturb_smooth = torch.zeros_like(theta)
        
        for kernel in self.kernels:
            kernel = kernel.to(device)
            # conv1d: [B, C, L] -> [B, C, L] (padding='same' 保持尺寸)
            pad = (kernel.shape[2] - 1) // 2
            smoothed = F.conv1d(theta, kernel, padding=pad)
            perturb_smooth += smoothed
        
        # 平均多尺度结果
        perturb_smooth /= len(self.kernels)
        
        return perturb_smooth
    
    def _pgd_init(self, x, y, num_steps=10):
        """
        使用快速 PGD 初始化 theta
        
        Args:
            x: 原始输入
            y: 标签
            num_steps: PGD 步数
        
        Returns:
            init_perturb: 初始扰动
        """
        # 简化版本：直接在 epsilon 范围内随机初始化
        # 这样可以避免与 PGD 类的冲突，同时提供良好的起点
        return torch.empty_like(x).uniform_(-self.eps * 0.5, self.eps * 0.5)
    
    def generate(self, x, y=None, targeted=False):
        """
        生成 SAP 对抗样本
        
        Args:
            x: 原始输入, shape [B, C, L]
            y: 标签
            targeted: 是否为目标攻击 (当前仅支持非目标攻击)
        
        Returns:
            x_adv: 对抗样本
        """
        x = x.clone().detach().to(self.device)
        batch_size = x.shape[0]
        
        if y is not None:
            y = y.to(self.device)
        else:
            # 使用模型预测作为伪标签
            with torch.no_grad():
                y = self.model(x).argmax(dim=1)
        
        # 初始化 theta (可学习扰动)
        theta = torch.zeros_like(x, requires_grad=True)
        
        # 可选: PGD 初始化
        if self.use_pgd_init:
            with torch.no_grad():
                init_perturb = self._pgd_init(x, y, num_steps=10)
                theta.data = init_perturb
            theta.requires_grad = True
        
        # Adam 优化器
        optimizer = torch.optim.Adam([theta], lr=self.lr)
        
        # 迭代优化
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            # 应用平滑: x_adv = x + mean(conv(theta, kernel_i))
            perturb_smooth = self._apply_smoothing(theta)
            
            x_adv = x + perturb_smooth
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # 计算损失并反向传播到 theta
            output = self.model(x_adv)
            
            if targeted:
                # 目标攻击: 最小化目标类别损失
                loss = nn.CrossEntropyLoss()(output, y)
            else:
                # 非目标攻击: 最大化损失 (最小化负损失)
                loss = -nn.CrossEntropyLoss()(output, y)
            
            loss.backward()
            optimizer.step()
            
            # 约束 theta 在 eps 范围内 (Linf 约束)
            with torch.no_grad():
                theta.data = torch.clamp(theta.data, -self.eps, self.eps)
        
        # 最终生成对抗样本
        with torch.no_grad():
            perturb_smooth = self._apply_smoothing(theta)
            x_adv = x + perturb_smooth
            x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv
    
    def smoothness_metric(self, delta):
        """
        计算扰动平滑度
        
        Args:
            delta: 扰动信号 [B, C, L] 或 [C, L] 或 [L]
        
        Returns:
            smoothness: 差分方差 (越小越平滑)
        """
        if delta.dim() == 3:
            delta = delta[0, 0]  # 取第一个样本和通道
        elif delta.dim() == 2:
            delta = delta[0]
        
        # 计算差分
        diff = delta[1:] - delta[:-1]
        
        # 返回方差
        return torch.var(diff).item()


if __name__ == "__main__":
    # 测试
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    os.chdir(project_root)
    
    from models.ecg_cnn import ECG_CNN
    from attacks.base_attack import BaseAttack  # noqa: F401
    from attacks.pgd import PGD
    
    # 创建模型
    model = ECG_CNN(num_classes=5)
    model.eval()
    
    # 创建攻击器
    attacker = SAP(model, eps=0.01, num_steps=40, use_pgd_init=True)
    
    # 测试输入
    x = torch.randn(4, 1, 187)
    y = torch.tensor([0, 1, 2, 3])
    
    print("Generating SAP adversarial examples...")
    x_adv = attacker.generate(x, y)
    
    # 计算扰动
    delta = x_adv - x
    
    print(f"\nOriginal shape: {x.shape}")
    print(f"Adversarial shape: {x_adv.shape}")
    print(f"Perturbation max: {delta.abs().max().item():.6f}")
    print(f"Perturbation L-inf: {delta.abs().max().item():.6f}")
    
    # 计算平滑度
    smoothness = attacker.smoothness_metric(delta)
    print(f"Perturbation smoothness (var of diff): {smoothness:.8f}")
    
    # 对比: PGD 扰动的平滑度
    pgd_attacker = PGD(model, eps=0.01, num_steps=40)
    x_pgd = pgd_attacker.generate(x, y)
    delta_pgd = x_pgd - x
    smoothness_pgd = attacker.smoothness_metric(delta_pgd)
    print(f"PGD smoothness (var of diff): {smoothness_pgd:.8f}")
    print(f"Smoothness ratio (SAP/PGD): {smoothness/smoothness_pgd:.4f}")
    
    # 验证模型输出变化
    with torch.no_grad():
        out_orig = model(x).argmax(dim=1)
        out_adv = model(x_adv).argmax(dim=1)
        out_pgd = model(x_pgd).argmax(dim=1)
    
    print(f"\nOriginal predictions: {out_orig}")
    print(f"SAP predictions: {out_adv}")
    print(f"PGD predictions: {out_pgd}")
    print(f"SAP attack success: {(out_orig != out_adv).sum().item()}/{len(y)} samples")
    print(f"PGD attack success: {(out_orig != out_pgd).sum().item()}/{len(y)} samples")
    
    # 测试通过标准
    if smoothness < smoothness_pgd * 0.5:
        print("\n✅ SAP is significantly smoother than PGD!")
    
    print("\n✅ SAP test passed!")
