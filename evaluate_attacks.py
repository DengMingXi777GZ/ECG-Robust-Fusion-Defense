"""
攻击效果评估脚本
对比不同攻击方法的指标
"""
import torch
import os
import sys
import argparse
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ecg_cnn import ECG_CNN
from data.mitbih_loader import get_mitbih_loaders
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from attacks.sap import SAP
from evaluation.attack_metrics import evaluate_attack, print_metrics
from evaluation.visualizer import (
    plot_waveform_comparison, 
    plot_attack_strength_curve,
    plot_frequency_spectrum,
    plot_metrics_table
)


def main(args):
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 检查模型是否存在
    if not os.path.exists(args.checkpoint):
        print(f"\n⚠️  Model checkpoint not found: {args.checkpoint}")
        print("Please train the model first using: python train_baseline.py")
        return
    
    # 加载模型
    print(f"\nLoading model from {args.checkpoint}...")
    model = ECG_CNN(num_classes=5).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded (test acc: {checkpoint.get('test_acc', 'N/A')}%)")
    
    # 加载测试数据
    print("\nLoading test data...")
    _, test_loader = get_mitbih_loaders(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size
    )
    
    # 获取一批数据用于评估
    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    
    print(f"Evaluation batch: {x_batch.shape}, labels: {y_batch.shape}")
    
    # 创建攻击实例
    print(f"\nInitializing attacks (eps={args.eps})...")
    attacks = {}
    
    if args.eval_fgsm:
        attacks['FGSM'] = FGSM(model, device=device, eps=args.eps)
    if args.eval_pgd:
        attacks['PGD'] = PGD(model, device=device, eps=args.eps, 
                             num_steps=args.pgd_steps, random_start=True)
    if args.eval_sap:
        attacks['SAP'] = SAP(model, device=device, eps=args.eps,
                             num_steps=args.sap_steps, use_pgd_init=True)
    
    # 评估每种攻击
    print(f"\n{'='*60}")
    print("Evaluating Attacks...")
    print(f"{'='*60}")
    
    metrics_dict = {}
    x_adv_dict = {}
    
    # Clean (baseline)
    with torch.no_grad():
        outputs = model(x_batch)
        clean_acc = (outputs.argmax(dim=1) == y_batch).float().mean().item() * 100
    print(f"\nClean Accuracy: {clean_acc:.2f}%")
    
    # 每种攻击
    for name, attack in attacks.items():
        print(f"\n{'-'*40}")
        print(f"Evaluating {name}...")
        print(f"{'-'*40}")
        
        # 生成对抗样本
        x_adv = attack.generate(x_batch, y_batch)
        x_adv_dict[name] = x_adv
        
        # 计算指标
        metrics = evaluate_attack(model, x_batch, x_adv, y_batch, device)
        metrics_dict[name] = metrics
        print_metrics(metrics, name)
    
    # 保存指标到 JSON
    if args.save_metrics:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / 'attack_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")
    
    # 生成可视化
    if args.visualize:
        print(f"\n{'='*60}")
        print("Generating Visualizations...")
        print(f"{'='*60}")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 波形对比图
        if 'SAP' in x_adv_dict:
            print("\nGenerating waveform comparison (SAP)...")
            plot_waveform_comparison(
                x_batch[0:1], x_adv_dict['SAP'][0:1],
                title="SAP Attack: Waveform Comparison",
                save_path=output_dir / 'figures' / 'attack_waveform_sap.png'
            )
        
        if 'PGD' in x_adv_dict:
            print("Generating waveform comparison (PGD)...")
            plot_waveform_comparison(
                x_batch[0:1], x_adv_dict['PGD'][0:1],
                title="PGD Attack: Waveform Comparison",
                save_path=output_dir / 'figures' / 'attack_waveform_pgd.png'
            )
        
        # 2. 指标表格
        print("Generating metrics table...")
        plot_metrics_table(
            metrics_dict,
            save_path=output_dir / 'figures' / 'attack_metrics_table.png'
        )
        
        # 3. 频谱分析 (对比 PGD 和 SAP)
        if 'PGD' in x_adv_dict and 'SAP' in x_adv_dict:
            print("Generating frequency spectrum analysis...")
            plot_frequency_spectrum(
                x_batch[0:1],
                x_adv_dict['PGD'][0:1],
                x_adv_dict['SAP'][0:1],
                save_path=output_dir / 'figures' / 'attack_frequency_analysis.png'
            )
        
        # 4. 攻击强度曲线
        if args.plot_strength_curve:
            print("Generating attack strength curve...")
            epsilons = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
            plot_attack_strength_curve(
                model, test_loader, attacks, epsilons, device,
                save_path=output_dir / 'figures' / 'attack_strength_curve.png'
            )
        
        print(f"\nAll figures saved to {output_dir / 'figures'}")
    
    # 打印总结
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Attack':<10} {'ASR (%)':<12} {'L2':<12} {'Smoothness':<15}")
    print(f"{'-'*60}")
    for name, metrics in metrics_dict.items():
        print(f"{name:<10} {metrics['ASR (%)']:<12.2f} {metrics['L2']:<12.6f} {metrics['Smoothness']:<15.8f}")
    print(f"{'='*60}")
    
    # 验证 SAP 的平滑度优势
    if 'SAP' in metrics_dict and 'PGD' in metrics_dict:
        sap_smooth = metrics_dict['SAP']['Smoothness']
        pgd_smooth = metrics_dict['PGD']['Smoothness']
        ratio = sap_smooth / pgd_smooth if pgd_smooth > 0 else float('inf')
        print(f"\nSAP smoothness / PGD smoothness ratio: {ratio:.4f}")
        if ratio < 0.1:
            print("✅ SAP is significantly smoother than PGD!")
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate adversarial attacks')
    
    # 数据路径
    parser.add_argument('--train-csv', type=str, default='data/mitbih_train.csv')
    parser.add_argument('--test-csv', type=str, default='data/mitbih_test.csv')
    
    # 模型路径
    parser.add_argument('--checkpoint', type=str, default='checkpoints/clean_model.pth')
    
    # 输出路径
    parser.add_argument('--output-dir', type=str, default='results')
    
    # 评估控制
    parser.add_argument('--eval-fgsm', action='store_true', default=True)
    parser.add_argument('--eval-pgd', action='store_true', default=True)
    parser.add_argument('--eval-sap', action='store_true', default=True)
    parser.add_argument('--skip-fgsm', action='store_true')
    parser.add_argument('--skip-pgd', action='store_true')
    parser.add_argument('--skip-sap', action='store_true')
    
    # 攻击参数
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--pgd-steps', type=int, default=20)
    parser.add_argument('--sap-steps', type=int, default=40)
    
    # 可视化控制
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--no-visualize', action='store_false', dest='visualize')
    parser.add_argument('--plot-strength-curve', action='store_true', default=False)
    parser.add_argument('--save-metrics', action='store_true', default=True)
    
    # 其他
    parser.add_argument('--batch-size', type=int, default=128)
    
    args = parser.parse_args()
    
    # 处理跳过选项
    if args.skip_fgsm:
        args.eval_fgsm = False
    if args.skip_pgd:
        args.eval_pgd = False
    if args.skip_sap:
        args.eval_sap = False
    
    main(args)
