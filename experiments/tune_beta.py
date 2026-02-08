"""
NSR超参数调优脚本 (Beta Tuning)
Task 2.8: 自动化搜索最佳beta值

搜索策略:
    - Beta候选值: [0.2, 0.4, 0.6, 0.8, 1.0]
    - 评估指标: ACC_robust = sqrt(ACC_clean * AUC_under_attack)
    - 选择验证集上ACC_robust最高的beta

输出:
    - results/beta_tuning_results.json: 所有实验结果
    - results/best_beta.txt: 最佳beta值
"""
import torch
import sys
import os
import argparse
import json
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense.train_nsr import train_nsr, evaluate_clean, evaluate_robust, compute_acc_robust
from models.ecg_cnn import ECG_CNN
from data.mitbih_loader import get_mitbih_loaders
from attacks import FGSM, PGD, SAP


def quick_evaluate_model(model, test_loader, attacks, device='cuda'):
    """
    快速评估模型（用于超参数搜索）
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        attacks: 攻击方法字典
        device: 计算设备
    
    Returns:
        results: 各攻击下的准确率
        acc_robust: 鲁棒性指标
    """
    model.eval()
    results = {}
    
    for attack_name, attacker in attacks.items():
        correct = 0
        total = 0
        
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            if attacker is not None:
                x_adv = attacker.generate(x, y)
            else:
                x_adv = x
            
            with torch.no_grad():
                outputs = model(x_adv)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        results[attack_name] = 100. * correct / total
    
    # 计算ACC_robust
    adv_accs = [results[name] for name in results.keys() if name != 'Clean']
    acc_robust = compute_acc_robust(results['Clean'], adv_accs)
    
    return results, acc_robust


def train_with_beta(
    beta,
    eps=0.05,
    epochs=30,  # 超参数搜索时使用较少epoch
    batch_size=256,
    lr=0.001,
    warmup_epochs=5,
    device='cuda',
    verbose=True
):
    """
    使用指定beta训练NSR模型
    
    Args:
        beta: NSR正则化系数
        eps: 扰动预算
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        warmup_epochs: warmup轮数
        device: 计算设备
        verbose: 是否输出详细信息
    
    Returns:
        model: 训练好的模型
        final_results: 最终评估结果
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training with Beta = {beta}")
        print(f"{'='*60}")
    
    # 加载数据
    train_loader, test_loader = get_mitbih_loaders(
        train_csv='data/mitbih_train.csv',
        test_csv='data/mitbih_test.csv',
        batch_size=batch_size
    )
    
    # 创建模型
    model = ECG_CNN(num_classes=5).to(device)
    
    # 导入NSRLoss和优化器
    from defense.nsr_loss import NSRLoss
    criterion = NSRLoss(beta=beta, eps=eps, num_classes=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 创建评估攻击
    attacks = {
        'Clean': None,
        'FGSM': FGSM(model, device=device, eps=eps),
        'PGD-20': PGD(model, device=device, eps=eps, num_steps=20, alpha=eps/4),
        'SAP': SAP(model, device=device, eps=eps, num_steps=40)
    }
    
    # 训练历史
    history = {
        'epochs': [],
        'train_loss': [],
        'test_clean_acc': [],
        'test_robust_acc': [],
        'acc_robust': []
    }
    
    best_acc_robust = 0
    best_state = None
    
    # 训练循环
    for epoch in range(epochs):
        # 延迟启动NSR
        if epoch < warmup_epochs:
            beta_current = 0.0
        else:
            beta_current = beta
        
        # 训练阶段
        model.train()
        total_loss = 0
        
        from tqdm import tqdm
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            
            # 临时设置beta
            original_beta = criterion.beta
            criterion.beta = beta_current
            loss, loss_dict = criterion(model, x, y, output)
            criterion.beta = original_beta
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss_dict['total']
        
        avg_loss = total_loss / len(train_loader)
        
        # 评估阶段（每5个epoch或最后）
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            results, acc_robust = quick_evaluate_model(model, test_loader, attacks, device)
            
            history['epochs'].append(epoch + 1)
            history['train_loss'].append(avg_loss)
            history['test_clean_acc'].append(results['Clean'])
            history['test_robust_acc'].append(results['PGD-20'])
            history['acc_robust'].append(acc_robust)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Clean={results['Clean']:.2f}%, "
                      f"PGD-20={results['PGD-20']:.2f}%, "
                      f"ACC_robust={acc_robust:.4f}")
            
            # 保存最佳模型
            if acc_robust > best_acc_robust:
                best_acc_robust = acc_robust
                best_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'results': results,
                    'acc_robust': acc_robust
                }
    
    # 加载最佳状态
    if best_state:
        model.load_state_dict(best_state['model_state_dict'])
        final_results = {
            'beta': beta,
            'best_epoch': best_state['epoch'],
            'final_results': best_state['results'],
            'acc_robust': best_state['acc_robust'],
            'history': history
        }
    else:
        results, acc_robust = quick_evaluate_model(model, test_loader, attacks, device)
        final_results = {
            'beta': beta,
            'best_epoch': epochs - 1,
            'final_results': results,
            'acc_robust': acc_robust,
            'history': history
        }
    
    return model, final_results


def tune_beta(
    betas=[0.2, 0.4, 0.6, 0.8, 1.0],
    eps=0.05,
    epochs=30,
    device='cuda',
    output_dir='results'
):
    """
    超参数调优主函数
    
    Args:
        betas: beta候选值列表
        eps: 扰动预算
        epochs: 每个beta的训练轮数
        device: 计算设备
        output_dir: 输出目录
    
    Returns:
        all_results: 所有实验结果
        best_beta: 最佳beta值
    """
    print(f"\n{'='*70}")
    print(f"NSR Hyperparameter Tuning: Beta Search")
    print(f"{'='*70}")
    print(f"Beta candidates: {betas}")
    print(f"Epsilon: {eps}")
    print(f"Epochs per beta: {epochs}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # 检查数据
    if not os.path.exists('data/mitbih_train.csv'):
        print("⚠️  Data not found! Please download MIT-BIH dataset first.")
        return None, None
    
    all_results = {}
    best_beta = None
    best_acc_robust = 0
    
    # 遍历所有beta值
    for i, beta in enumerate(betas):
        print(f"\n{'='*70}")
        print(f"Experiment {i+1}/{len(betas)}: Beta = {beta}")
        print(f"{'='*70}")
        
        # 训练
        model, results = train_with_beta(
            beta=beta,
            eps=eps,
            epochs=epochs,
            device=device,
            verbose=True
        )
        
        all_results[f'beta_{beta}'] = results
        
        # 更新最佳beta
        if results['acc_robust'] > best_acc_robust:
            best_acc_robust = results['acc_robust']
            best_beta = beta
            
            # 保存最佳模型
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'beta': beta,
                'acc_robust': best_acc_robust,
                'model_state_dict': model.state_dict(),
                'results': results['final_results']
            }, 'checkpoints/nsr_best_beta.pth')
    
    # 汇总结果
    print(f"\n{'='*70}")
    print(f"Tuning Summary")
    print(f"{'='*70}")
    
    summary_data = []
    for beta in betas:
        key = f'beta_{beta}'
        results = all_results[key]
        summary_data.append({
            'Beta': beta,
            'Clean Acc (%)': f"{results['final_results']['Clean']:.2f}",
            'FGSM Acc (%)': f"{results['final_results']['FGSM']:.2f}",
            'PGD-20 Acc (%)': f"{results['final_results']['PGD-20']:.2f}",
            'SAP Acc (%)': f"{results['final_results']['SAP']:.2f}",
            'ACC_robust': f"{results['acc_robust']:.4f}"
        })
    
    # 打印表格
    import pandas as pd
    from tabulate import tabulate
    
    df = pd.DataFrame(summary_data)
    print("\n" + tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    print(f"\n{'='*70}")
    print(f"Best Beta: {best_beta} (ACC_robust = {best_acc_robust:.4f})")
    print(f"{'='*70}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存所有结果
    results_path = os.path.join(output_dir, 'beta_tuning_results.json')
    
    # 转换numpy类型以便JSON序列化
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    serializable_results['best_beta'] = best_beta
    serializable_results['best_acc_robust'] = best_acc_robust
    serializable_results['timestamp'] = datetime.now().isoformat()
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # 保存最佳beta值
    best_beta_path = os.path.join(output_dir, 'best_beta.txt')
    with open(best_beta_path, 'w') as f:
        f.write(f"Best Beta: {best_beta}\n")
        f.write(f"ACC_robust: {best_acc_robust:.4f}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    
    print(f"Best beta saved to: {best_beta_path}")
    
    # 保存CSV摘要
    csv_path = os.path.join(output_dir, 'beta_tuning_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to: {csv_path}")
    
    return all_results, best_beta


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='NSR Beta Hyperparameter Tuning')
    
    parser.add_argument('--betas', type=float, nargs='+', 
                        default=[0.2, 0.4, 0.6, 0.8, 1.0],
                        help='Beta candidates')
    parser.add_argument('--eps', type=float, default=0.05, help='Epsilon')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Epochs per beta (use fewer for faster search)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    tune_beta(
        betas=args.betas,
        eps=args.eps,
        epochs=args.epochs,
        device=device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
