"""
Layer 2 防御层统一执行脚本
运行完整的防御层流程：从生成对抗样本到训练防御模型

使用方法:
    # 完整流程
    python run_layer2_defense.py --full
    
    # 仅生成对抗样本
    python run_layer2_defense.py --generate-adv
    
    # 仅训练标准AT
    python run_layer2_defense.py --train-at
    
    # 仅训练NSR
    python run_layer2_defense.py --train-nsr --beta 0.4
    
    # 仅训练AT+NSR
    python run_layer2_defense.py --train-at-nsr --beta 0.4
    
    # 超参数调优
    python run_layer2_defense.py --tune-beta
    
    # 评估所有模型
    python run_layer2_defense.py --evaluate
"""
import torch
import os
import sys
import argparse
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def log(message):
    """打印带时间戳的日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def check_prerequisites():
    """检查前置条件"""
    log("Checking prerequisites...")
    
    # 检查模型
    if not os.path.exists('checkpoints/clean_model.pth'):
        log("❌ Clean model not found at checkpoints/clean_model.pth")
        log("Please train the baseline model first: python train_baseline.py")
        return False
    
    # 检查数据
    if not os.path.exists('data/mitbih_train.csv') or not os.path.exists('data/mitbih_test.csv'):
        log("❌ MIT-BIH dataset not found in data/")
        log("Please download the dataset first")
        return False
    
    log("✅ All prerequisites satisfied")
    return True


def generate_adversarial_samples(eps=0.05):
    """生成对抗样本"""
    log(f"Generating adversarial samples (eps={eps})...")
    
    cmd = [
        'python', 'generate_adversarial_dataset.py',
        '--checkpoint', 'checkpoints/clean_model.pth',
        '--eps', str(eps),
        '--pgd-steps', '40',
        '--sap-steps', '40',
        '--output-dir', f'data/adversarial/eps{int(eps*100):02d}/',
        '--batch-size', '128'
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        log("✅ Adversarial samples generated successfully")
        return True
    else:
        log("❌ Failed to generate adversarial samples")
        return False


def train_standard_at(eps=0.05, epochs=50):
    """训练标准对抗训练模型"""
    log("Training Standard Adversarial Training model...")
    
    cmd = [
        'python', 'defense/train_standard_at.py',
        '--eps', str(eps),
        '--epochs', str(epochs),
        '--batch-size', '256',
        '--checkpoint', 'checkpoints/adv_standard_at.pth'
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        log("✅ Standard AT model trained successfully")
        return True
    else:
        log("❌ Failed to train Standard AT model")
        return False


def train_nsr(beta=0.4, eps=0.05, epochs=50):
    """训练NSR模型"""
    log(f"Training NSR model (beta={beta})...")
    
    cmd = [
        'python', 'defense/train_nsr.py',
        '--beta', str(beta),
        '--eps', str(eps),
        '--epochs', str(epochs),
        '--batch-size', '256',
        '--checkpoint', f'checkpoints/nsr_beta{beta}.pth'
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        log(f"✅ NSR model (beta={beta}) trained successfully")
        return True
    else:
        log("❌ Failed to train NSR model")
        return False


def train_at_nsr(beta=0.4, eps=0.05, epochs=50):
    """训练AT+NSR联合模型"""
    log(f"Training AT+NSR combined model (beta={beta})...")
    
    cmd = [
        'python', 'defense/train_at_nsr.py',
        '--beta', str(beta),
        '--eps', str(eps),
        '--epochs', str(epochs),
        '--batch-size', '256',
        '--checkpoint', 'checkpoints/at_nsr.pth'
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        log("✅ AT+NSR combined model trained successfully")
        return True
    else:
        log("❌ Failed to train AT+NSR model")
        return False


def tune_beta(eps=0.05, epochs=30):
    """超参数调优"""
    log("Running hyperparameter tuning for beta...")
    
    cmd = [
        'python', 'experiments/tune_beta.py',
        '--betas', '0.2', '0.4', '0.6', '0.8', '1.0',
        '--eps', str(eps),
        '--epochs', str(epochs)
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        log("✅ Hyperparameter tuning completed")
        return True
    else:
        log("❌ Failed to complete hyperparameter tuning")
        return False


def evaluate_models(eps=0.05):
    """评估所有模型"""
    log("Evaluating all defense models...")
    
    cmd = [
        'python', 'evaluation/defense_eval.py',
        '--compare',
        '--eps', str(eps),
        '--output', 'results/defense_evaluation.json'
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        log("✅ Evaluation completed")
        return True
    else:
        log("❌ Failed to evaluate models")
        return False


def main():
    parser = argparse.ArgumentParser(description='Layer 2 Defense Pipeline')
    
    # 主要操作
    parser.add_argument('--full', action='store_true', 
                        help='Run full pipeline: generate -> tune -> train -> evaluate')
    parser.add_argument('--generate-adv', action='store_true',
                        help='Generate adversarial samples only')
    parser.add_argument('--train-at', action='store_true',
                        help='Train Standard AT model only')
    parser.add_argument('--train-nsr', action='store_true',
                        help='Train NSR model only')
    parser.add_argument('--train-at-nsr', action='store_true',
                        help='Train AT+NSR combined model only')
    parser.add_argument('--tune-beta', action='store_true',
                        help='Run beta hyperparameter tuning only')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate all models only')
    
    # 参数
    parser.add_argument('--eps', type=float, default=0.05, help='Epsilon for adversarial training')
    parser.add_argument('--beta', type=float, default=0.4, help='NSR beta coefficient')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--tune-epochs', type=int, default=30, help='Epochs for tuning')
    parser.add_argument('--skip-checks', action='store_true', help='Skip prerequisite checks')
    
    args = parser.parse_args()
    
    # 检查前置条件
    if not args.skip_checks:
        if not check_prerequisites():
            return
    
    log("="*70)
    log("Layer 2 Defense Pipeline Starting")
    log("="*70)
    log(f"Epsilon: {args.eps}")
    log(f"Beta: {args.beta}")
    log(f"Epochs: {args.epochs}")
    log(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    log("="*70)
    
    success = True
    
    # 完整流程
    if args.full:
        log("\n>>> Running FULL pipeline")
        success = success and generate_adversarial_samples(args.eps)
        success = success and tune_beta(args.eps, args.tune_epochs)
        success = success and train_standard_at(args.eps, args.epochs)
        success = success and train_nsr(args.beta, args.eps, args.epochs)
        success = success and train_at_nsr(args.beta, args.eps, args.epochs)
        success = success and evaluate_models(args.eps)
    
    # 单独操作
    else:
        if args.generate_adv:
            success = success and generate_adversarial_samples(args.eps)
        
        if args.tune_beta:
            success = success and tune_beta(args.eps, args.tune_epochs)
        
        if args.train_at:
            success = success and train_standard_at(args.eps, args.epochs)
        
        if args.train_nsr:
            success = success and train_nsr(args.beta, args.eps, args.epochs)
        
        if args.train_at_nsr:
            success = success and train_at_nsr(args.beta, args.eps, args.epochs)
        
        if args.evaluate:
            success = success and evaluate_models(args.eps)
    
    # 总结
    log("="*70)
    if success:
        log("✅ All operations completed successfully!")
    else:
        log("⚠️  Some operations failed. Check logs above.")
    log("="*70)


if __name__ == "__main__":
    main()
