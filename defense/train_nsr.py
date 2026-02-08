"""
NSR训练管道 (NSR Regularization Training)
Task 2.5: 使用NSRLoss替代CrossEntropy进行训练

关键特性:
    - 使用NSRLoss替代标准交叉熵
    - 延迟启动NSR（前10epoch只用MSE，防止初期梯度爆炸）
    - 支持超参数搜索（beta: [0.2, 0.4, 0.6, 0.8, 1.0]）

评估指标:
    ACC_robust = sqrt(ACC_clean * AUC_under_attack)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ecg_cnn import ECG_CNN
from data.mitbih_loader import get_mitbih_loaders
from attacks import PGD, FGSM, SAP
from defense.nsr_loss import NSRLoss


def train_epoch_nsr(model, loader, optimizer, criterion, beta_current, device='cuda'):
    """
    NSR训练的一个epoch
    
    Args:
        model: 模型
        loader: 数据加载器
        optimizer: 优化器
        criterion: NSRLoss实例
        beta_current: 当前beta值（用于延迟启动）
        device: 计算设备
    
    Returns:
        avg_loss: 平均损失
        avg_acc: 平均准确率
        loss_components: 损失组件字典
    """
    model.train()
    total_loss = 0
    total_mse = 0
    total_margin = 0
    total_nsr = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training NSR")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        # 前向传播
        output = model(x)
        
        # 计算NSR损失（临时设置beta）
        original_beta = criterion.beta
        criterion.beta = beta_current
        loss, loss_dict = criterion(model, x, y, output)
        criterion.beta = original_beta
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss_dict['total']
        total_mse += loss_dict['mse']
        total_margin += loss_dict['margin']
        total_nsr += loss_dict['nsr']
        
        _, predicted = output.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss_dict["total"]:.4f}',
            'mse': f'{loss_dict["mse"]:.4f}',
            'nsr': f'{loss_dict["nsr"]:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    n_batches = len(loader)
    return (
        total_loss / n_batches,
        100. * correct / total,
        {
            'mse': total_mse / n_batches,
            'margin': total_margin / n_batches,
            'nsr': total_nsr / n_batches
        }
    )


def evaluate_clean(model, loader, device='cuda'):
    """评估干净样本准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval Clean"):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    return 100. * correct / total


def evaluate_robust(model, loader, attacks, device='cuda'):
    """
    评估对抗鲁棒性（多攻击）
    
    Args:
        model: 模型
        loader: 数据加载器
        attacks: 攻击方法字典 {'name': attacker}
        device: 计算设备
    
    Returns:
        results: 各攻击下的准确率字典
    """
    model.eval()
    results = {}
    
    for attack_name, attacker in attacks.items():
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Eval {attack_name}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            # 生成对抗样本
            if attacker is not None:
                x_adv = attacker.generate(x, y)
            else:
                x_adv = x
            
            # 评估
            with torch.no_grad():
                outputs = model(x_adv)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            
            pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
        
        results[attack_name] = 100. * correct / total
    
    return results


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


def train_nsr(
    beta=0.4,
    eps=0.05,
    epochs=50,
    batch_size=256,
    lr=0.001,
    warmup_epochs=10,
    checkpoint_path='checkpoints/nsr_beta0.4.pth',
    device='cuda'
):
    """
    NSR训练主函数
    
    Args:
        beta: NSR正则化系数
        eps: 扰动预算
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        warmup_epochs: warmup轮数（只用MSE，关闭NSR）
        checkpoint_path: 模型保存路径
        device: 计算设备
    
    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    print(f"{'='*60}")
    print(f"NSR Regularization Training")
    print(f"{'='*60}")
    print(f"Beta: {beta}")
    print(f"Epsilon: {eps}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 检查数据文件
    train_csv = 'data/mitbih_train.csv'
    test_csv = 'data/mitbih_test.csv'
    
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print(f"\n⚠️  Data files not found!")
        return None, None
    
    # 加载数据
    print("Loading data...")
    train_loader, test_loader = get_mitbih_loaders(
        train_csv=train_csv,
        test_csv=test_csv,
        batch_size=batch_size
    )
    
    # 创建模型
    model = ECG_CNN(num_classes=5).to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # NSR损失（beta会在训练过程中动态调整）
    criterion = NSRLoss(beta=beta, eps=eps, num_classes=5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # 创建评估攻击
    attacks = {
        'FGSM': FGSM(model, device=device, eps=eps),
        'PGD-20': PGD(model, device=device, eps=eps, num_steps=20, alpha=eps/4),
        'SAP': SAP(model, device=device, eps=eps, num_steps=40)
    }
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_clean_acc': [],
        'test_robust': {},
        'acc_robust': [],
        'epochs': [],
        'beta_history': []
    }
    
    # 最佳模型
    best_acc_robust = 0
    
    # 训练循环
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # 延迟启动NSR
        if epoch < warmup_epochs:
            beta_current = 0.0  # 关闭NSR
            print(f"[Warmup] Beta = {beta_current} (MSE only)")
        else:
            beta_current = beta  # 开启NSR
            print(f"[Training] Beta = {beta_current}")
        
        # 训练
        train_loss, train_acc, loss_comps = train_epoch_nsr(
            model, train_loader, optimizer, criterion, beta_current, device
        )
        
        # 评估
        test_clean_acc = evaluate_clean(model, test_loader, device)
        test_robust = evaluate_robust(model, test_loader, attacks, device)
        
        # 计算ACC_robust
        adv_accs = [test_robust[name] for name in ['FGSM', 'PGD-20', 'SAP']]
        acc_robust = compute_acc_robust(test_clean_acc, adv_accs)
        
        # 学习率调度
        scheduler.step(test_clean_acc)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_clean_acc'].append(test_clean_acc)
        history['test_robust'] = test_robust
        history['acc_robust'].append(acc_robust)
        history['epochs'].append(epoch + 1)
        history['beta_history'].append(beta_current)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (MSE: {loss_comps['mse']:.4f}, NSR: {loss_comps['nsr']:.4f})")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Test Clean Acc: {test_clean_acc:.2f}%")
        print(f"  Test Robust Acc:")
        for name, acc in test_robust.items():
            print(f"    - {name}: {acc:.2f}%")
        print(f"  ACC_robust: {acc_robust:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型（基于ACC_robust）
        if acc_robust > best_acc_robust:
            best_acc_robust = acc_robust
            
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_clean_acc': test_clean_acc,
                'test_robust': test_robust,
                'acc_robust': acc_robust,
                'beta': beta,
                'eps': eps,
            }, checkpoint_path)
            print(f"\n[OK] Saved best model (ACC_robust: {acc_robust:.4f})")
    
    # 训练完成总结
    print(f"\n{'='*60}")
    print(f"Training Completed!")
    print(f"{'='*60}")
    print(f"Best ACC_robust: {best_acc_robust:.4f}")
    print(f"Model saved to: {checkpoint_path}")
    
    # 验收检查
    print(f"\n{'='*60}")
    print(f"Acceptance Criteria Check:")
    print(f"{'='*60}")
    if best_acc_robust >= 0.75:
        print(f"✅ ACC_robust: {best_acc_robust:.4f} >= 0.75")
    else:
        print(f"❌ ACC_robust: {best_acc_robust:.4f} < 0.75")
    
    # 保存训练历史
    history_path = checkpoint_path.replace('.pth', '_history.json')
    # 转换numpy类型为Python类型以便JSON序列化
    history_serializable = {
        k: [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in v] 
        if isinstance(v, list) else v 
        for k, v in history.items()
    }
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NSR Regularization Training')
    
    parser.add_argument('--beta', type=float, default=0.4, help='NSR regularization coefficient')
    parser.add_argument('--eps', type=float, default=0.05, help='Epsilon')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/nsr_beta0.4.pth',
                        help='Checkpoint save path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 根据beta值修改checkpoint路径
    checkpoint_path = args.checkpoint
    if args.beta != 0.4:
        checkpoint_path = checkpoint_path.replace('beta0.4', f'beta{args.beta}')
    
    train_nsr(
        beta=args.beta,
        eps=args.eps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        checkpoint_path=checkpoint_path,
        device=device
    )
