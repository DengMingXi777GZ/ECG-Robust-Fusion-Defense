"""
对抗攻击基类
定义标准接口和通用功能
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseAttack(ABC):
    """
    对抗攻击基类
    
    所有攻击方法都应继承此类并实现 generate 方法
    """
    
    def __init__(self, model, device='cpu', eps=0.01):
        """
        Args:
            model: 目标模型 (PyTorch nn.Module)
            device: 计算设备
            eps: 扰动上限 (L-infinity norm)
        """
        self.model = model
        self.device = device
        self.eps = eps
        self.model.to(device)
        self.model.eval()  # 确保模型在评估模式
    
    @abstractmethod
    def generate(self, x, y=None, targeted=False):
        """
        生成对抗样本
        
        Args:
            x: 原始输入, shape [B, C, L]
            y: 真实标签 (非目标攻击) 或目标标签 (目标攻击)
            targeted: 是否为目标攻击
        
        Returns:
            x_adv: 对抗样本, 与 x 同 shape
        """
        pass
    
    def clip(self, x_adv, x_orig):
        """
        投影回 epsilon 球和 [0, 1] 范围
        
        Args:
            x_adv: 对抗样本
            x_orig: 原始样本
        
        Returns:
            裁剪后的对抗样本
        """
        # 投影回 epsilon 球 (L-infinity)
        x_adv = torch.clamp(x_adv, x_orig - self.eps, x_orig + self.eps)
        # 投影回 [0, 1] 范围
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv
    
    def compute_gradient(self, x, y):
        """
        计算损失函数对输入的梯度
        
        Args:
            x: 输入样本, requires_grad=True
            y: 标签
        
        Returns:
            gradient: 梯度张量
        """
        x_copy = x.clone().detach().requires_grad_(True)
        
        output = self.model(x_copy)
        loss = nn.CrossEntropyLoss()(output, y)
        
        self.model.zero_grad()
        loss.backward()
        
        return x_copy.grad.data
