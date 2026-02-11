"""
训练 ECG 基线模型
目标: 测试集准确率 >= 91%
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ecg_cnn import ECG_CNN #从models目录下的ecg_cnn.py文件中导入ECG_CNN类
from data.mitbih_loader import get_mitbih_loaders


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training") #创建一个进度条来显示训练过程
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1) #outputs.max(1)返回每行的最大值和对应的索引，predicted是索引，即预测的类别,1是按行取最大值
        total += y.size(0) #y.size(0)返回当前批次的样本数量，即总的样本数
        correct += predicted.eq(y).sum().item() #predicted.eq(y)返回一个布尔张量，表示预测是否正确，sum().item()计算正确的数量
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    acc = 100. * correct / total
    return total_loss / len(loader), acc


def train_model(epochs=30, batch_size=128, lr=0.001, checkpoint_path='checkpoints/clean_model.pth'):
    """
    训练基线模型
    
    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        checkpoint_path: 模型保存路径
    """
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 检查数据文件是否存在
    train_csv = 'data/mitbih_train.csv'
    test_csv = 'data/mitbih_test.csv'
    
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print(f"\n⚠️  Data files not found!")
        print(f"Expected: {train_csv} and {test_csv}")
        print("\nPlease download MIT-BIH dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/shayanfazeli/heartbeat")
        print("\nAnd place the CSV files in the 'data/' directory.")
        return None
    
    # 数据加载
    print("\nLoading data...")
    train_loader, test_loader = get_mitbih_loaders(
        train_csv=train_csv,
        test_csv=test_csv,
        batch_size=batch_size
    )
    
    # 模型
    model = ECG_CNN(num_classes=5).to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    # 学习率调整器会在验证准确率不提升时降低学习率，factor=0.5表示每次降低一半，patience=5表示等待5个epoch后再调整
    
    # 训练
    best_acc = 0
    print(f"\n{'='*50}")
    print(f"Training for {epochs} epochs...")
    print(f"{'='*50}\n")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, checkpoint_path)
            print(f"[OK] Saved best model (acc: {best_acc:.2f}%)")
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    
    # 验证是否达到目标
    if best_acc >= 91.0:
        print(f"[OK] Target achieved! (>= 91%)")
    else:
        print(f"[WARNING] Target not achieved. Try more epochs or adjust hyperparameters.")
    
    return model, best_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ECG baseline model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/clean_model.pth',
                        help='Checkpoint save path')
    
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=args.checkpoint
    )
