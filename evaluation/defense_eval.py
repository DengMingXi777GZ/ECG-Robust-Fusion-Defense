"""
鲁棒性评估框架 (Defense Evaluation Framework)
Task 2.7: 对比Clean模型 vs 防御模型在相同攻击下的表现

评估标准:
    - Clean: 无攻击
    - FGSM: 快速梯度攻击
    - PGD-20: 20步PGD攻击
    - PGD-100: 强攻击（100步）
    - SAP: 平滑对抗扰动

对比实验:
    - Clean Model (Layer 1基线)
    - Standard AT (Madry's AT)
    - NSR (β=0.4)
    - AT+NSR (融合方案)
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from tabulate import tabulate

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ecg_cnn import ECG_CNN
from data.mitbih_loader import get_mitbih_loaders
from attacks import FGSM, PGD, SAP


def evaluate_robustness(model, test_loader, eps=0.05, device='cuda'):
    """
    评估模型在多攻击下的鲁棒性
    
    Args:
        model: 待评估模型
        test_loader: 测试数据加载器
        eps: 扰动预算
        device: 计算设备
    
    Returns:
        results: 各攻击下的准确率字典
        details: 详细结果
    """
    model.eval()
    
    # 定义攻击方法（使用Layer 1的attacks，eps=0.05）
    attacks = {
        'Clean': None,
        'FGSM': FGSM(model, device=device, eps=eps),
        'PGD-20': PGD(model, device=device, eps=eps, num_steps=20, alpha=eps/4),
        'PGD-100': PGD(model, device=device, eps=eps, num_steps=100, alpha=eps/4),
        'SAP': SAP(model, device=device, eps=eps, num_steps=40)
    }
    
    results = {}
    all_predictions = {}
    
    for attack_name, attacker in attacks.items():
        correct = 0
        total = 0
        predictions = []
        labels = []
        
        pbar = tqdm(test_loader, desc=f"Eval {attack_name}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            # 生成对抗样本
            if attacker is not None:
                x_adv = attacker.generate(x, y)
            else:
                x_adv = x
            
            # 评估
            with torch.no_grad():
                output = model(x_adv)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
                predictions.extend(pred.cpu().numpy())
                labels.extend(y.cpu().numpy())
            
            pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
        
        accuracy = 100.0 * correct / total
        results[attack_name] = accuracy
        all_predictions[attack_name] = {
            'predictions': predictions,
            'labels': labels
        }
    
    return results, all_predictions


def compute_acc_robust(clean_acc, adv_accuracies):
    """
    计算ACC_robust（基于Ma论文）
    
    公式: ACC_robust = sqrt(ACC_clean * mean(adv_accuracies))
    
    Args:
        clean_acc: 干净样本准确率 (%)
        adv_accuracies: 对抗样本准确率列表 (%)
    
    Returns:
        acc_robust: 鲁棒性指标
    """
    clean_acc_norm = clean_acc / 100.0
    adv_accuracies_norm = [acc / 100.0 for acc in adv_accuracies]
    mean_adv_acc = np.mean(adv_accuracies_norm)
    
    return np.sqrt(clean_acc_norm * mean_adv_acc)


def evaluate_single_model(model_path, test_loader, eps=0.05, device='cuda', model_name='Model'):
    """
    评估单个模型
    
    Args:
        model_path: 模型路径
        test_loader: 测试数据加载器
        eps: 扰动预算
        device: 计算设备
        model_name: 模型名称
    
    Returns:
        results: 评估结果
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        return None
    
    # 加载模型
    model = ECG_CNN(num_classes=5).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"⚠️  Model saved on GPU but running on CPU, retrying with CPU map_location...")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        else:
            raise e
    except Exception as e:
        print(f"⚠️  Failed to load model: {e}")
        return None
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully")
    
    # 评估鲁棒性
    results, details = evaluate_robustness(model, test_loader, eps, device)
    
    # 计算ACC_robust
    adv_accs = [results[name] for name in results.keys() if name != 'Clean']
    acc_robust = compute_acc_robust(results['Clean'], adv_accs)
    results['ACC_robust'] = acc_robust
    
    print(f"\nResults for {model_name}:")
    for attack, acc in results.items():
        print(f"  {attack}: {acc:.2f}%")
    
    return results


def compare_models(model_paths, test_loader, eps=0.05, device='cuda'):
    """
    对比多个模型的鲁棒性
    
    Args:
        model_paths: 模型路径字典 {'name': 'path'}
        test_loader: 测试数据加载器
        eps: 扰动预算
        device: 计算设备
    
    Returns:
        comparison_df: 对比表格
    """
    all_results = {}
    
    for model_name, model_path in model_paths.items():
        results = evaluate_single_model(model_path, test_loader, eps, device, model_name)
        if results is not None:
            all_results[model_name] = results
    
    # 创建对比表格
    if all_results:
        # 转换为DataFrame
        df = pd.DataFrame(all_results).T
        
        # 确保列顺序
        columns = ['Clean', 'FGSM', 'PGD-20', 'PGD-100', 'SAP', 'ACC_robust']
        df = df[[col for col in columns if col in df.columns]]
        
        print(f"\n{'='*80}")
        print("Robustness Comparison Table")
        print(f"{'='*80}")
        print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.2f'))
        
        return df
    
    return None


def evaluate_with_epsilon_curve(model_path, test_loader, epsilons, device='cuda'):
    """
    评估模型在不同epsilon下的鲁棒性（绘制Accuracy vs Epsilon曲线）
    
    Args:
        model_path: 模型路径
        test_loader: 测试数据加载器
        epsilons: epsilon值列表
        device: 计算设备
    
    Returns:
        curve_results: 各epsilon下的准确率
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Epsilon Curve")
    print(f"{'='*60}")
    
    # 加载模型
    model = ECG_CNN(num_classes=5).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"⚠️  Failed to load model: {e}")
        return None
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    curve_results = {
        'epsilon': epsilons,
        'FGSM': [],
        'PGD-20': [],
        'PGD-100': [],
        'SAP': []
    }
    
    for eps in epsilons:
        print(f"\nEvaluating with eps={eps:.3f}")
        
        # 创建攻击器
        attacks = {
            'FGSM': FGSM(model, device=device, eps=eps),
            'PGD-20': PGD(model, device=device, eps=eps, num_steps=20, alpha=eps/4),
            'PGD-100': PGD(model, device=device, eps=eps, num_steps=100, alpha=eps/4),
            'SAP': SAP(model, device=device, eps=eps, num_steps=40)
        }
        
        for attack_name, attacker in attacks.items():
            correct = 0
            total = 0
            
            for x, y in tqdm(test_loader, desc=f"{attack_name} (eps={eps:.3f})", leave=False):
                x, y = x.to(device), y.to(device)
                x_adv = attacker.generate(x, y)
                
                with torch.no_grad():
                    output = model(x_adv)
                    pred = output.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            
            accuracy = 100.0 * correct / total
            curve_results[attack_name].append(accuracy)
            print(f"  {attack_name}: {accuracy:.2f}%")
    
    return curve_results


def save_results(results, output_path='results/defense_evaluation.json'):
    """保存评估结果到JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换numpy类型
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_serializable[key] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in value.items()
            }
        elif isinstance(value, pd.DataFrame):
            results_serializable[key] = value.to_dict()
        else:
            results_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Defense Evaluation Framework')
    
    parser.add_argument('--model', type=str, default=None, help='Single model path to evaluate')
    parser.add_argument('--model-name', type=str, default='Model', help='Model name')
    parser.add_argument('--compare', action='store_true', help='Compare all defense models')
    parser.add_argument('--epsilon-curve', action='store_true', help='Evaluate epsilon curve')
    parser.add_argument('--eps', type=float, default=0.05, help='Epsilon for adversarial attacks')
    parser.add_argument('--eps-max', type=float, default=0.10, help='Max epsilon for curve')
    parser.add_argument('--eps-steps', type=int, default=11, help='Number of epsilon steps')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--output', type=str, default='results/defense_evaluation.json',
                        help='Output path for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("\nLoading data...")
    _, test_loader = get_mitbih_loaders(
        train_csv='data/mitbih_train.csv',
        test_csv='data/mitbih_test.csv',
        batch_size=args.batch_size
    )
    
    results = {}
    
    # 单模型评估
    if args.model:
        single_result = evaluate_single_model(
            args.model, test_loader, args.eps, device, args.model_name
        )
        if single_result:
            results[args.model_name] = single_result
    
    # 对比评估
    if args.compare:
        model_paths = {
            'Clean (Layer 1)': 'checkpoints/clean_model.pth',
            'Standard AT': 'checkpoints/adv_standard_at.pth',
            'NSR (β=0.4)': 'checkpoints/nsr_beta0.4.pth',
            'AT+NSR': 'checkpoints/at_nsr.pth'
        }
        
        comparison_df = compare_models(model_paths, test_loader, args.eps, device)
        if comparison_df is not None:
            results['comparison'] = comparison_df
            
            # 同时保存CSV
            csv_path = 'results/defense_comparison.csv'
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            comparison_df.to_csv(csv_path)
            print(f"\nComparison table saved to: {csv_path}")
    
    # Epsilon曲线评估
    if args.epsilon_curve and args.model:
        epsilons = np.linspace(0, args.eps_max, args.eps_steps)
        curve_results = evaluate_with_epsilon_curve(
            args.model, test_loader, epsilons, device
        )
        results['epsilon_curve'] = curve_results
        
        # 保存曲线数据
        curve_path = 'results/epsilon_curve.json'
        with open(curve_path, 'w') as f:
            json.dump(curve_results, f, indent=2)
        print(f"\nEpsilon curve saved to: {curve_path}")
    
    # 保存结果
    if results:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
