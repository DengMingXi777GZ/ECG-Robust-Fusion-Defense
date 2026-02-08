"""
标准对抗训练 (Standard Adversarial Training)
Task 2.3: Madry's AT 实现

参考: "Towards Deep Learning Models Resistant to Adversarial Attacks" (Madry et al., ICLR 2018)

核心逻辑:
    - Min-Max优化: min_θ E[max_δ L(f(x+δ), y)]
    - 混合Clean和Adv样本 (各50%)
    - 训练时PGD使用10步（弱攻击，防止过拟合）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import argparse
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ecg_cnn import ECG_CNN
from data.mitbih_loader import get_mitbih_loaders
from attacks import PGD


def train_epoch_at(model, loader, optimizer, criterion, eps=0.05, device='cuda'):
    """
    标准对抗训练的一个epoch
    
    Args:
        model: 模型
        loader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        eps: 扰动预算 (默认0.05，基于Layer 1结论)
        device: 计算设备
    
    Returns:
        avg_loss: 平均损失
        avg_acc: 平均准确率
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 创建攻击器（训练时使用10步，比评估时的40步弱）
    attacker = PGD(model, device=device, eps=eps, num_steps=10, 
                   alpha=eps/4, random_start=True)
    
    pbar = tqdm(loader, desc="Training AT")
    for x_clean, y in pbar:
        x_clean, y = x_clean.to(device), y.to(device)
        
        # 1. 生成对抗样本（使用当前模型状态）
        x_adv = attacker.generate(x_clean, y)
        
        # 2. 混合数据（Madry标准做法：各50%）
        x_mixed = torch.cat([x_clean, x_adv], dim=0)
        y_mixed = torch.cat([y, y], dim=0)
        
        # 3. 前向传播
        output = model(x_mixed)
        loss = criterion(output, y_mixed)
        
        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += y_mixed.size(0)
        correct += predicted.eq(y_mixed).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total


def evaluate_clean(model, loader, criterion, device='cuda'):
    """评估干净样本准确率"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval Clean"):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def evaluate_robust(model, loader, eps=0.05, steps=20, device='cuda'):
    """
    评估对抗鲁棒性 (PGD-20攻击)
    
    Args:
        model: 模型
        loader: 数据加载器
        eps: 扰动预算
        steps: PGD步数（评估用20步）
        device: 计算设备
    
    Returns:
        robust_acc: 鲁棒准确率
    """
    model.eval()
    correct = 0
    total = 0
    
    # 创建攻击器（评估用20步）
    attacker = PGD(model, device=device, eps=eps, num_steps=steps, 
                   alpha=eps/4, random_start=True)
    
    pbar = tqdm(loader, desc=f"Eval Robust (PGD-{steps})")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        # 生成对抗样本
        x_adv = attacker.generate(x, y)
        
        # 评估
        with torch.no_grad():
            outputs = model(x_adv)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        pbar.set_postfix({'robust_acc': f'{100.*correct/total:.2f}%'})
    
    return 100. * correct / total


def train_standard_at(
    epochs=50,
    batch_size=256,
    lr=0.001,
    eps=0.05,
    warmup_epochs=5,
    checkpoint_path='checkpoints/adv_standard_at.pth',
    device='cuda'
):
    """
    标准对抗训练主函数
    
    超参数配置（基于Ma & Liang 2022）:
        - Epochs: 50
        - Optimizer: Adam, lr=0.001
        - Epsilon: 0.05（基于Layer 1结论）
        - Attack steps: 10（训练时）
        - Batch size: 256
        - Warmup: 前5epoch只用clean数据
    
    验收标准:
        - Clean Accuracy ≥ 88%（允许比93.43%下降5%以内）
        - PGD-20 (eps=0.05) Accuracy ≥ 60%（鲁棒性提升）
    
    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        eps: 扰动预算
        warmup_epochs: warmup轮数（只用clean数据）
        checkpoint_path: 模型保存路径
        device: 计算设备
    
    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    print(f"{'='*60}")
    print(f"Standard Adversarial Training (Madry's AT)")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epsilon: {eps}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 检查数据文件
    train_csv = 'data/mitbih_train.csv'
    test_csv = 'data/mitbih_test.csv'
    
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print(f"\n⚠️  Data files not found!")
        print(f"Expected: {train_csv} and {test_csv}")
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
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_clean_acc': [],
        'test_robust_acc': [],
        'epochs': []
    }
    
    # 最佳模型
    best_robust_acc = 0
    best_clean_acc = 0
    
    # 训练循环
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        if epoch < warmup_epochs:
            # Warmup阶段：只用clean数据训练
            print(f"[Warmup] Training on clean data only...")
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            for x, y in tqdm(train_loader, desc="Warmup Training"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            train_loss = total_loss / len(train_loader)
            train_acc = 100. * correct / total
        else:
            # 对抗训练阶段
            train_loss, train_acc = train_epoch_at(
                model, train_loader, optimizer, criterion, eps, device
            )
        
        # 评估
        _, test_clean_acc = evaluate_clean(model, test_loader, criterion, device)
        test_robust_acc = evaluate_robust(model, test_loader, eps, steps=20, device=device)
        
        # 学习率调度
        scheduler.step(test_clean_acc)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_clean_acc'].append(test_clean_acc)
        history['test_robust_acc'].append(test_robust_acc)
        history['epochs'].append(epoch + 1)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Test Clean Acc: {test_clean_acc:.2f}%")
        print(f"  Test Robust Acc (PGD-20): {test_robust_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型（基于鲁棒准确率）
        if test_robust_acc > best_robust_acc:
            best_robust_acc = test_robust_acc
            best_clean_acc = test_clean_acc
            
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_clean_acc': test_clean_acc,
                'test_robust_acc': test_robust_acc,
                'eps': eps,
            }, checkpoint_path)
            print(f"\n[OK] Saved best model (clean: {test_clean_acc:.2f}%, robust: {test_robust_acc:.2f}%)")
    
    # 训练完成总结
    print(f"\n{'='*60}")
    print(f"Training Completed!")
    print(f"{'='*60}")
    print(f"Best Clean Accuracy: {best_clean_acc:.2f}%")
    print(f"Best Robust Accuracy (PGD-20): {best_robust_acc:.2f}%")
    print(f"Model saved to: {checkpoint_path}")
    
    # 验收检查
    print(f"\n{'='*60}")
    print(f"Acceptance Criteria Check:")
    print(f"{'='*60}")
    if best_clean_acc >= 88.0:
        print(f"✅ Clean Accuracy: {best_clean_acc:.2f}% >= 88%")
    else:
        print(f"❌ Clean Accuracy: {best_clean_acc:.2f}% < 88%")
    
    if best_robust_acc >= 60.0:
        print(f"✅ Robust Accuracy (PGD-20): {best_robust_acc:.2f}% >= 60%")
    else:
        print(f"❌ Robust Accuracy (PGD-20): {best_robust_acc:.2f}% < 60%")
    
    # 保存训练历史
    history_path = checkpoint_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standard Adversarial Training')
    
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--eps', type=float, default=0.05, help='Epsilon for adversarial training')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs with clean data')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/adv_standard_at.pth',
                        help='Checkpoint save path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    train_standard_at(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eps=args.eps,
        warmup_epochs=args.warmup_epochs,
        checkpoint_path=args.checkpoint,
        device=device
    )
