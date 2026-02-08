"""
详细调试梯度计算和攻击过程
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.ecg_cnn import ECG_CNN
from data.mitbih_loader import MITBIHDataset

def debug_gradient_flow():
    """调试梯度流"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model = ECG_CNN(num_classes=5).to(device)
    checkpoint = torch.load('checkpoints/clean_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 注意：eval模式下梯度计算仍然有效
    
    # 加载一个样本
    dataset = MITBIHDataset('data/mitbih_test.csv', transform='normalize', preload=True)
    x, y = dataset[0]
    x = x.unsqueeze(0).to(device)  # [1, 1, 187]
    y = torch.tensor([y]).to(device)
    
    print(f"\n样本信息: shape={x.shape}, label={y.item()}")
    
    # 前向传播
    x.requires_grad = True
    output = model(x)
    loss = F.cross_entropy(output, y)
    
    print(f"\n原始预测: class={output.argmax().item()}")
    print(f"原始概率分布: {torch.softmax(output, dim=1).squeeze()}")
    print(f"原始损失: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    grad = x.grad
    print(f"\n梯度统计:")
    print(f"  梯度shape: {grad.shape}")
    print(f"  梯度min: {grad.min().item():.6f}")
    print(f"  梯度max: {grad.max().item():.6f}")
    print(f"  梯度mean: {grad.mean().item():.6f}")
    print(f"  梯度abs mean: {grad.abs().mean().item():.6f}")
    print(f"  正梯度数量: {(grad > 0).sum().item()}")
    print(f"  负梯度数量: {(grad < 0).sum().item()}")
    print(f"  零梯度数量: {(grad == 0).sum().item()}")
    
    # 手动执行FGSM攻击
    eps = 0.03
    grad_sign = grad.sign()
    
    print(f"\nFGSM攻击 (eps={eps}):")
    
    # 非目标攻击：+ grad_sign (最大化损失)
    x_adv = x + eps * grad_sign
    x_adv = torch.clamp(x_adv, 0, 1)
    
    with torch.no_grad():
        output_adv = model(x_adv)
        loss_adv = F.cross_entropy(output_adv, y)
        print(f"  新预测: class={output_adv.argmax().item()}")
        print(f"  新概率分布: {torch.softmax(output_adv, dim=1).squeeze()}")
        print(f"  新损失: {loss_adv.item():.6f}")
        print(f"  损失变化: {loss_adv.item() - loss.item():.6f}")
        print(f"  攻击成功: {output.argmax().item() != output_adv.argmax().item()}")
    
    # 检查相反方向（目标攻击）
    print(f"\n如果方向相反（目标攻击）:")
    x_adv_wrong = x - eps * grad_sign
    x_adv_wrong = torch.clamp(x_adv_wrong, 0, 1)
    
    with torch.no_grad():
        output_wrong = model(x_adv_wrong)
        loss_wrong = F.cross_entropy(output_wrong, y)
        print(f"  新预测: class={output_wrong.argmax().item()}")
        print(f"  新概率分布: {torch.softmax(output_wrong, dim=1).squeeze()}")
        print(f"  新损失: {loss_wrong.item():.6f}")
        print(f"  损失变化: {loss_wrong.item() - loss.item():.6f}")
        print(f"  攻击成功: {output.argmax().item() != output_wrong.argmax().item()}")
    
    # 测试不同eps
    print(f"\n不同epsilon的攻击效果:")
    for eps in [0.01, 0.03, 0.05, 0.1]:
        x_adv = x + eps * grad_sign
        x_adv = torch.clamp(x_adv, 0, 1)
        with torch.no_grad():
            pred_adv = model(x_adv).argmax().item()
            orig_pred = output.argmax().item()
            success = orig_pred != pred_adv
        print(f"  eps={eps:.2f}: {orig_pred} -> {pred_adv}, success={success}")

def check_model_gradient_mode():
    """检查模型各层的梯度状态"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECG_CNN(num_classes=5).to(device)
    checkpoint = torch.load('checkpoints/clean_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n模型梯度状态检查:")
    print(f"model.training: {model.training}")
    
    # 检查各层的requires_grad
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"  WARNING: {name} requires_grad=False")
    
    # 检查BatchNorm状态
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            print(f"  {name}: training={module.training}, track_running_stats={module.track_running_stats}")

if __name__ == "__main__":
    check_model_gradient_mode()
    debug_gradient_flow()
