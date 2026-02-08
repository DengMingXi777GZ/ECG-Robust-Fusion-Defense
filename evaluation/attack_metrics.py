"""
攻击评估指标计算器
"""
import torch
import numpy as np


def attack_success_rate(model, x_adv, y_true, device='cpu'):
    """
    计算攻击成功率 (ASR)
    
    ASR = 被误分类的对抗样本比例
    
    Args:
        model: 目标模型
        x_adv: 对抗样本 [B, C, L]
        y_true: 真实标签 [B]
    
    Returns:
        asr: 攻击成功率 (0-100%)
    """
    model.eval()
    x_adv = x_adv.to(device)
    y_true = y_true.to(device)
    
    with torch.no_grad():
        outputs = model(x_adv)
        predictions = outputs.argmax(dim=1)
        success = (predictions != y_true).sum().item()
    
    asr = 100.0 * success / len(y_true)
    return asr


def perturbation_l2(x_adv, x_orig):
    """
    计算 L2 扰动 (归一化)
    
    L2 = ||x_adv - x||_2 / sqrt(dim)
    
    Args:
        x_adv: 对抗样本 [B, C, L]
        x_orig: 原始样本 [B, C, L]
    
    Returns:
        l2: 平均 L2 扰动
    """
    delta = x_adv - x_orig
    batch_size = delta.shape[0]
    dim = delta.shape[1] * delta.shape[2]  # C * L
    
    # 计算每个样本的 L2 范数
    l2_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
    
    # 归一化
    l2_normalized = l2_norms / np.sqrt(dim)
    
    return l2_normalized.mean().item()


def perturbation_linf(x_adv, x_orig):
    """
    计算 L-infinity 扰动
    
    Linf = max(|x_adv - x|)
    
    Args:
        x_adv: 对抗样本 [B, C, L]
        x_orig: 原始样本 [B, C, L]
    
    Returns:
        linf: 平均 L-infinity 扰动
    """
    delta = (x_adv - x_orig).abs()
    linf = delta.view(delta.shape[0], -1).max(dim=1)[0]
    return linf.mean().item()


def signal_noise_ratio(x_orig, delta):
    """
    计算信噪比 (SNR)
    
    SNR = 20 * log10(std(x) / std(delta))
    
    Args:
        x_orig: 原始样本 [B, C, L]
        delta: 扰动 [B, C, L]
    
    Returns:
        snr: 平均 SNR (dB)
    """
    batch_size = x_orig.shape[0]
    
    # 计算每个样本的 std
    x_std = x_orig.view(batch_size, -1).std(dim=1)
    delta_std = delta.view(batch_size, -1).std(dim=1)
    
    # 避免除零
    delta_std = torch.clamp(delta_std, min=1e-10)
    
    # SNR
    snr = 20 * torch.log10(x_std / delta_std)
    
    return snr.mean().item()


def smoothness(delta):
    """
    计算扰动平滑度
    
    平滑度 = var(diff(delta))
    值越小表示越平滑
    
    Args:
        delta: 扰动 [B, C, L]
    
    Returns:
        smooth: 平均平滑度
    """
    batch_size = delta.shape[0]
    channels = delta.shape[1]
    length = delta.shape[2]
    
    # 计算差分 [B, C, L-1]
    diff = delta[:, :, 1:] - delta[:, :, :-1]
    
    # 计算每个样本的方差
    smooth_vals = diff.view(batch_size, -1).var(dim=1)
    
    return smooth_vals.mean().item()


def evaluate_attack(model, x_orig, x_adv, y_true, device='cpu'):
    """
    综合评估攻击效果
    
    Returns:
        metrics: 字典包含所有指标
    """
    x_orig = x_orig.to(device)
    x_adv = x_adv.to(device)
    y_true = y_true.to(device)
    delta = x_adv - x_orig
    
    metrics = {
        'ASR (%)': attack_success_rate(model, x_adv, y_true, device),
        'L2': perturbation_l2(x_adv, x_orig),
        'Linf': perturbation_linf(x_adv, x_orig),
        'SNR (dB)': signal_noise_ratio(x_orig, delta),
        'Smoothness': smoothness(delta),
    }
    
    return metrics


def print_metrics(metrics, name="Attack"):
    """打印指标"""
    print(f"\n{'='*50}")
    print(f"{name} Metrics")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"  {key:20s}: {value:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    # 测试
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.ecg_cnn import ECG_CNN
    from attacks.fgsm import FGSM
    
    # 创建模型
    model = ECG_CNN(num_classes=5)
    model.eval()
    
    # 测试数据
    x = torch.randn(8, 1, 187)
    y = torch.randint(0, 5, (8,))
    
    # 生成对抗样本
    attacker = FGSM(model, eps=0.01)
    x_adv = attacker.generate(x, y)
    
    # 评估
    metrics = evaluate_attack(model, x, x_adv, y)
    print_metrics(metrics, "FGSM")
    
    print("\n[OK] Metrics test passed!")
