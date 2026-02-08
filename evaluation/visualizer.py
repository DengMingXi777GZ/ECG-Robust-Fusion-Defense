"""
攻击可视化工具
"""
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def plot_waveform_comparison(x_orig, x_adv, title="Waveform Comparison", save_path=None):
    """
    绘制原始波形与对抗波形对比图 (参考 Han 论文 Fig.1)
    
    Args:
        x_orig: 原始样本 [1, C, L] 或 [C, L] 或 [L]
        x_adv: 对抗样本 [1, C, L] 或 [C, L] 或 [L]
        title: 图表标题
        save_path: 保存路径
    """
    # 统一处理输入
    if x_orig.dim() == 3:
        x_orig = x_orig[0, 0]
        x_adv = x_adv[0, 0]
    elif x_orig.dim() == 2:
        x_orig = x_orig[0]
        x_adv = x_adv[0]
    
    x_orig = x_orig.cpu().numpy()
    x_adv = x_adv.cpu().numpy()
    delta = x_adv - x_orig
    
    # 创建子图
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=150)
    
    time_axis = np.arange(len(x_orig))
    
    # 子图1: 波形对比
    ax1 = axes[0]
    ax1.plot(time_axis, x_orig, 'b-', label='Original ECG', linewidth=1.5, alpha=0.8)
    ax1.plot(time_axis, x_adv, 'r-', label='Adversarial ECG', linewidth=1.5, alpha=0.6)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(x_orig))
    
    # 子图2: 扰动波形
    ax2 = axes[1]
    ax2.plot(time_axis, delta, 'g-', linewidth=1.2)
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Perturbation', fontsize=12)
    ax2.set_title('Adversarial Perturbation (δ = x_adv - x)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(x_orig))
    
    # 添加平滑度信息
    diff = delta[1:] - delta[:-1]
    smoothness = np.var(diff)
    ax2.text(0.02, 0.95, f'Smoothness: {smoothness:.2e}', 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()


def plot_attack_strength_curve(model, test_loader, attacks_dict, epsilons, device='cpu', save_path=None):
    """
    绘制攻击强度曲线 (准确率 vs epsilon)
    
    Args:
        model: 目标模型
        test_loader: 测试数据加载器
        attacks_dict: 攻击方法字典 {'name': attack_instance}
        epsilons: epsilon 值列表
        device: 计算设备
        save_path: 保存路径
    """
    model.eval()
    
    # 存储结果
    results = {name: [] for name in attacks_dict.keys()}
    results['Clean'] = []
    
    # 获取一批测试数据
    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    
    # 对每个 epsilon 进行评估
    for eps in epsilons:
        print(f"Evaluating epsilon={eps:.4f}...")
        
        # Clean accuracy (只在第一次计算)
        if len(results['Clean']) == 0:
            with torch.no_grad():
                outputs = model(x_batch)
                clean_acc = (outputs.argmax(dim=1) == y_batch).float().mean().item() * 100
            results['Clean'] = [clean_acc] * len(epsilons)
        
        # 每种攻击
        for name, attack in attacks_dict.items():
            # 更新 epsilon
            attack.eps = eps
            
            # 生成对抗样本
            x_adv = attack.generate(x_batch, y_batch)
            
            # 计算准确率
            with torch.no_grad():
                outputs = model(x_adv)
                acc = (outputs.argmax(dim=1) == y_batch).float().mean().item() * 100
            
            results[name].append(acc)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # 颜色映射
    colors = {'Clean': 'black', 'FGSM': 'red', 'PGD': 'blue', 'SAP': 'green'}
    markers = {'Clean': 'o', 'FGSM': 's', 'PGD': '^', 'SAP': 'D'}
    
    for name, accs in results.items():
        ax.plot(epsilons, accs, 
                label=name, 
                color=colors.get(name, 'gray'),
                marker=markers.get(name, None),
                linewidth=2, markersize=6)
    
    ax.set_xlabel('Epsilon (ε)', fontsize=12)
    ax.set_ylabel('Model Accuracy (%)', fontsize=12)
    ax.set_title('Attack Strength vs Model Accuracy', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()
    
    return results


def plot_frequency_spectrum(x_orig, x_adv_pgd, x_adv_sap, save_path=None):
    """
    频谱分析图 (验证 SAP 的平滑性)
    
    Args:
        x_orig: 原始样本
        x_adv_pgd: PGD 对抗样本
        x_adv_sap: SAP 对抗样本
        save_path: 保存路径
    """
    # 统一处理输入
    def get_perturbation(x_adv, x_orig):
        if x_adv.dim() == 3:
            x_adv = x_adv[0, 0]
            x_orig = x_orig[0, 0]
        elif x_adv.dim() == 2:
            x_adv = x_adv[0]
            x_orig = x_orig[0]
        return (x_adv - x_orig).cpu().numpy()
    
    delta_pgd = get_perturbation(x_adv_pgd, x_orig)
    delta_sap = get_perturbation(x_adv_sap, x_orig)
    
    # FFT
    fft_pgd = np.abs(np.fft.fft(delta_pgd))
    fft_sap = np.abs(np.fft.fft(delta_sap))
    freqs = np.fft.fftfreq(len(delta_pgd))
    
    # 只取正频率
    pos_idx = freqs > 0
    freqs = freqs[pos_idx]
    fft_pgd = fft_pgd[pos_idx]
    fft_sap = fft_sap[pos_idx]
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    
    # 时域扰动
    axes[0, 0].plot(delta_pgd, 'b-', linewidth=1)
    axes[0, 0].set_title('PGD Perturbation (Time Domain)', fontsize=12)
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(delta_sap, 'g-', linewidth=1)
    axes[0, 1].set_title('SAP Perturbation (Time Domain)', fontsize=12)
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 频域
    axes[1, 0].semilogy(freqs, fft_pgd, 'b-', linewidth=1)
    axes[1, 0].set_title('PGD Perturbation (Frequency Domain)', fontsize=12)
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Magnitude (log)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(freqs, fft_sap, 'g-', linewidth=1)
    axes[1, 1].set_title('SAP Perturbation (Frequency Domain)', fontsize=12)
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].set_ylabel('Magnitude (log)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Frequency Spectrum Analysis: PGD vs SAP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()


def plot_metrics_table(metrics_dict, save_path=None):
    """
    绘制指标对比表格
    
    Args:
        metrics_dict: {attack_name: {metric_name: value}}
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    headers = ['Attack'] + list(list(metrics_dict.values())[0].keys())
    
    rows = []
    for name, metrics in metrics_dict.items():
        row = [name] + [f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()]
        rows.append(row)
    
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Attack Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # 测试
    import tempfile
    import os
    
    # 生成模拟数据
    x_orig = torch.randn(1, 1, 187)
    x_adv = x_orig + torch.randn(1, 1, 187) * 0.01
    
    # 测试波形对比图
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_waveform.png')
        plot_waveform_comparison(x_orig, x_adv, save_path=save_path)
        assert os.path.exists(save_path)
        print(f"[OK] Waveform comparison test passed!")
    
    print("\n[OK] All visualizer tests passed!")
