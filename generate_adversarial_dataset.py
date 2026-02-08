"""
对抗样本数据集生成器
为 Layer 2 (防御层) 生成对抗训练数据
"""
import torch
import os
import sys
import argparse
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ecg_cnn import ECG_CNN
from data.mitbih_loader import get_mitbih_loaders
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from attacks.sap import SAP


def generate_adversarial_dataset(model, test_loader, attack, device='cpu', desc="Generating"):
    """
    为整个测试集生成对抗样本
    
    Returns:
        x_adv_list: 对抗样本列表
        y_list: 标签列表
        x_orig_list: 原始样本列表
    """
    model.eval()
    
    x_adv_list = []
    y_list = []
    x_orig_list = []
    
    for x, y in tqdm(test_loader, desc=desc):
        x = x.to(device)
        y = y.to(device)
        
        # 生成对抗样本
        x_adv = attack.generate(x, y)
        
        x_adv_list.append(x_adv.cpu())
        y_list.append(y.cpu())
        x_orig_list.append(x.cpu())
    
    # 合并所有批次
    x_adv_all = torch.cat(x_adv_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    x_orig_all = torch.cat(x_orig_list, dim=0)
    
    return x_adv_all, y_all, x_orig_all


def main(args):
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 检查模型是否存在
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"\n⚠️  Model checkpoint not found: {checkpoint_path}")
        print("Please train the model first using: python train_baseline.py")
        return
    
    # 加载模型
    print(f"\nLoading model from {checkpoint_path}...")
    model = ECG_CNN(num_classes=5).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 生成 FGSM 对抗样本
    if args.generate_fgsm:
        print(f"\n{'='*50}")
        print(f"Generating FGSM adversarial examples (eps={args.eps})")
        print(f"{'='*50}")
        
        fgsm = FGSM(model, device=device, eps=args.eps)
        x_adv, y, x_orig = generate_adversarial_dataset(
            model, test_loader, fgsm, device, desc="FGSM"
        )
        
        # 保存
        save_path = output_dir / 'test_fgsm.pt'
        torch.save({
            'x_adv': x_adv,
            'y': y,
            'x_orig': x_orig,
            'eps': args.eps,
            'attack': 'FGSM'
        }, save_path)
        print(f"Saved to {save_path}")
        print(f"  Shape: x_adv={x_adv.shape}, y={y.shape}")
    
    # 2. 生成 PGD 对抗样本
    if args.generate_pgd:
        print(f"\n{'='*50}")
        print(f"Generating PGD adversarial examples (eps={args.eps}, steps={args.pgd_steps})")
        print(f"{'='*50}")
        
        pgd = PGD(model, device=device, eps=args.eps, 
                  num_steps=args.pgd_steps, random_start=True)
        x_adv, y, x_orig = generate_adversarial_dataset(
            model, test_loader, pgd, device, desc="PGD"
        )
        
        # 保存
        save_path = output_dir / 'test_pgd.pt'
        torch.save({
            'x_adv': x_adv,
            'y': y,
            'x_orig': x_orig,
            'eps': args.eps,
            'steps': args.pgd_steps,
            'attack': 'PGD'
        }, save_path)
        print(f"Saved to {save_path}")
        print(f"  Shape: x_adv={x_adv.shape}, y={y.shape}")
    
    # 3. 生成 SAP 对抗样本
    if args.generate_sap:
        print(f"\n{'='*50}")
        print(f"Generating SAP adversarial examples (eps={args.eps}, steps={args.sap_steps})")
        print(f"{'='*50}")
        
        sap = SAP(model, device=device, eps=args.eps,
                  num_steps=args.sap_steps, use_pgd_init=True)
        x_adv, y, x_orig = generate_adversarial_dataset(
            model, test_loader, sap, device, desc="SAP"
        )
        
        # 保存
        save_path = output_dir / 'test_sap.pt'
        torch.save({
            'x_adv': x_adv,
            'y': y,
            'x_orig': x_orig,
            'eps': args.eps,
            'steps': args.sap_steps,
            'attack': 'SAP'
        }, save_path)
        print(f"Saved to {save_path}")
        print(f"  Shape: x_adv={x_adv.shape}, y={y.shape}")
    
    print(f"\n{'='*50}")
    print("Adversarial dataset generation completed!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate adversarial dataset')
    
    # 数据路径
    parser.add_argument('--train-csv', type=str, default='data/mitbih_train.csv')
    parser.add_argument('--test-csv', type=str, default='data/mitbih_test.csv')
    
    # 模型路径
    parser.add_argument('--checkpoint', type=str, default='checkpoints/clean_model.pth')
    
    # 输出路径
    parser.add_argument('--output-dir', type=str, default='data/adversarial')
    
    # 攻击参数
    parser.add_argument('--eps', type=float, default=0.01, help='Epsilon (perturbation bound)')
    parser.add_argument('--pgd-steps', type=int, default=20, help='PGD iteration steps')
    parser.add_argument('--sap-steps', type=int, default=40, help='SAP optimization steps')
    
    # 生成控制
    parser.add_argument('--generate-fgsm', action='store_true', default=True)
    parser.add_argument('--generate-pgd', action='store_true', default=True)
    parser.add_argument('--generate-sap', action='store_true', default=True)
    parser.add_argument('--skip-fgsm', action='store_true', help='Skip FGSM generation')
    parser.add_argument('--skip-pgd', action='store_true', help='Skip PGD generation')
    parser.add_argument('--skip-sap', action='store_true', help='Skip SAP generation')
    
    # 其他
    parser.add_argument('--batch-size', type=int, default=128)
    
    args = parser.parse_args()
    
    # 处理跳过选项
    if args.skip_fgsm:
        args.generate_fgsm = False
    if args.skip_pgd:
        args.generate_pgd = False
    if args.skip_sap:
        args.generate_sap = False
    
    main(args)
