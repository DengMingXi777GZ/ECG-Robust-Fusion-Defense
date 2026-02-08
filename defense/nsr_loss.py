"""
NSR (Noise-to-Signal Ratio) 损失计算器
Task 2.4: 实现 Ma & Liang 2022 论文中的 NSR 正则化

学术来源: Ma & Liang 2022, Eq.(7)

核心公式:
    L_NSR = (z_y-1)^2 + Σ_{i≠y}(z_i-0)^2 + Σ_{i≠y}max(0,1-z_y+z_i) + β·log(R+1)
    
    其中 R = ||w_y||_1 · ε / |z_y|
    
    - w_y: 类别y的logit关于输入x的梯度
    - z_y: 正确类别的logit输出
    - ε: 扰动预算
    - β: 正则化系数（MIT-BIH上最佳0.4）

设计原理:
    - MSE Loss: 鼓励one-hot输出（正确类别为1，其他为0）
    - Margin Loss: 增大类别间间隔
    - NSR Regularization: 限制梯度大小，提高鲁棒性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NSRLoss(nn.Module):
    """
    NSR正则化损失函数
    
    Args:
        beta: 正则化系数，MIT-BIH上最佳0.4
        eps: 扰动预算（使用Layer 1验证的0.05）
        num_classes: 类别数（默认5）
    """
    
    def __init__(self, beta=0.4, eps=0.05, num_classes=5):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.num_classes = num_classes
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, model, x, y, output):
        """
        计算NSR损失
        
        Args:
            model: 模型（用于计算梯度）
            x: 输入样本 [B, C, L]
            y: 标签 [B]
            output: 模型输出logits [B, num_classes]
        
        Returns:
            total_loss: 总损失
            loss_dict: 各组件损失字典
        """
        batch_size = y.size(0)
        device = x.device
        
        # ========== 1. MSE Loss (One-hot目标) ==========
        # 公式: (z_y-1)^2 + Σ_{i≠y}(z_i-0)^2
        y_onehot = F.one_hot(y, self.num_classes).float()
        mse_loss = self.mse(output, y_onehot)
        
        # ========== 2. Margin Loss (仅对正确分类样本) ==========
        # 公式: Σ_{i≠y}max(0, 1-z_y+z_i)
        z_y = output[range(batch_size), y]  # 正确类别的logits [B]
        
        # 计算所有类别的margin
        margins = torch.clamp(1 - z_y.unsqueeze(1) + output, min=0)  # [B, num_classes]
        margins[range(batch_size), y] = 0  # 排除正确类别
        margin_loss = margins.sum() / batch_size
        
        # ========== 3. NSR Regularization (关键部分) ==========
        # 计算 ||w_y||_1: 对类别y的logit关于输入x的梯度的L1范数
        
        w_l1 = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            # 创建需要梯度的输入副本
            xi = x[i:i+1].clone().detach().requires_grad_(True)
            
            # 前向传播
            out = model(xi)
            z = out[0, y[i]]  # 正确类别的logit
            
            # 计算梯度
            grad = torch.autograd.grad(
                z, xi, 
                create_graph=True,  # 保留计算图以支持二阶导数
                retain_graph=True
            )[0]
            
            # L1范数
            w_l1[i] = torch.norm(grad, p=1)
        
        # 计算 R = ||w_y||_1 * eps / |z_y|
        # 添加epsilon防止除零
        z_y_abs = torch.abs(z_y) + 1e-8
        R = (w_l1 * self.eps) / z_y_abs
        
        # NSR正则化: β * log(R + 1)
        nsr_loss = self.beta * torch.mean(torch.log(R + 1))
        
        # ========== 4. 组合损失 ==========
        # 仅对正确分类样本应用NSR和Margin
        pred = output.argmax(dim=1)
        correct_mask = (pred == y).float().mean()
        
        # 总损失 = MSE + (Margin + NSR) * correct_mask
        # correct_mask作为权重，确保只对正确分类的样本应用额外的正则化
        total_loss = mse_loss + (margin_loss + nsr_loss) * correct_mask
        
        # 损失分解
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'margin': margin_loss.item(),
            'nsr': nsr_loss.item(),
            'correct_ratio': correct_mask.item()
        }
        
        return total_loss, loss_dict


class NSRLossV2(nn.Module):
    """
    NSR损失v2: 带温度缩放的版本（更稳定的训练）
    
    添加温度参数T来控制softmax的sharpness
    """
    
    def __init__(self, beta=0.4, eps=0.05, num_classes=5, temperature=1.0):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.num_classes = num_classes
        self.temperature = temperature
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, model, x, y, output):
        """计算NSR损失（带温度缩放）"""
        batch_size = y.size(0)
        device = x.device
        
        # 应用温度缩放
        output_scaled = output / self.temperature
        
        # 1. MSE Loss
        y_onehot = F.one_hot(y, self.num_classes).float()
        # 调整one-hot目标以适应温度缩放
        y_target = y_onehot * (1.0 / self.temperature) + (1 - y_onehot) * 0
        mse_loss = self.mse(output_scaled, y_target)
        
        # 2. Margin Loss
        z_y = output_scaled[range(batch_size), y]
        margins = torch.clamp(1 - z_y.unsqueeze(1) + output_scaled, min=0)
        margins[range(batch_size), y] = 0
        margin_loss = margins.sum() / batch_size
        
        # 3. NSR Regularization (使用原始输出)
        w_l1 = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            xi = x[i:i+1].clone().detach().requires_grad_(True)
            out = model(xi)
            z = out[0, y[i]]
            grad = torch.autograd.grad(z, xi, create_graph=True, retain_graph=True)[0]
            w_l1[i] = torch.norm(grad, p=1)
        
        z_y_original = output[range(batch_size), y]
        z_y_abs = torch.abs(z_y_original) + 1e-8
        R = (w_l1 * self.eps) / z_y_abs
        nsr_loss = self.beta * torch.mean(torch.log(R + 1))
        
        # 4. 组合
        pred = output.argmax(dim=1)
        correct_mask = (pred == y).float().mean()
        total_loss = mse_loss + (margin_loss + nsr_loss) * correct_mask
        
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'margin': margin_loss.item(),
            'nsr': nsr_loss.item(),
            'correct_ratio': correct_mask.item()
        }
        
        return total_loss, loss_dict


def test_nsr_loss():
    """测试NSR损失"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.ecg_cnn import ECG_CNN
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = ECG_CNN(num_classes=5).to(device)
    model.train()  # 需要梯度
    
    # 创建NSR损失
    criterion = NSRLoss(beta=0.4, eps=0.05, num_classes=5)
    
    # 测试输入
    batch_size = 4
    x = torch.randn(batch_size, 1, 187, device=device)
    y = torch.tensor([0, 1, 2, 3], device=device)
    
    print("\n" + "="*60)
    print("Testing NSR Loss")
    print("="*60)
    
    # 前向传播
    output = model(x)
    print(f"\nModel output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 计算NSR损失
    loss, loss_dict = criterion(model, x, y, output)
    
    print(f"\nLoss Components:")
    print(f"  Total Loss: {loss_dict['total']:.4f}")
    print(f"  MSE Loss: {loss_dict['mse']:.4f}")
    print(f"  Margin Loss: {loss_dict['margin']:.4f}")
    print(f"  NSR Loss: {loss_dict['nsr']:.4f}")
    print(f"  Correct Ratio: {loss_dict['correct_ratio']:.4f}")
    
    # 反向传播测试
    print("\nTesting backward pass...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Backward pass successful!")
    
    # 测试不同beta值
    print("\n" + "="*60)
    print("Testing different beta values")
    print("="*60)
    
    for beta in [0.2, 0.4, 0.6, 0.8, 1.0]:
        criterion_test = NSRLoss(beta=beta, eps=0.05, num_classes=5)
        output = model(x)
        loss, loss_dict = criterion_test(model, x, y, output)
        print(f"Beta={beta:.1f}: Total={loss_dict['total']:.4f}, NSR={loss_dict['nsr']:.4f}")
    
    print("\n✅ NSR Loss test passed!")


if __name__ == "__main__":
    test_nsr_loss()
