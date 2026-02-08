"""
调试攻击代码，检查ASR计算是否正确
"""
import torch
import torch.nn as nn
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.ecg_cnn import ECG_CNN
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from attacks.sap import SAP
from data.mitbih_loader import MITBIHDataset

def debug_single_sample():
    """调试单样本攻击"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model = ECG_CNN(num_classes=5).to(device)
    checkpoint = torch.load('checkpoints/clean_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载数据集
    dataset = MITBIHDataset('data/mitbih_test.csv', transform='normalize', preload=True)
    
    # 找一个模型能正确分类的样本
    print("\n[1] 寻找模型能正确分类的样本...")
    for i in range(100):
        x, y = dataset[i]
        x_batch = x.unsqueeze(0).to(device)  # [1, 1, 187]
        y_batch = torch.tensor([y]).to(device)
        
        with torch.no_grad():
            pred = model(x_batch).argmax(dim=1).item()
            prob = torch.softmax(model(x_batch), dim=1).max().item()
        
        if pred == y and prob > 0.8:  # 找一个高置信度的正确样本
            print(f"  Sample {i}: True={y}, Pred={pred}, Prob={prob:.4f}")
            break
    
    # 测试各种攻击
    print("\n[2] 测试 FGSM 攻击...")
    fgsm = FGSM(model, device=device, eps=0.03)  # 增大eps便于测试
    x_adv = fgsm.generate(x_batch, y_batch, targeted=False)
    
    with torch.no_grad():
        orig_pred = model(x_batch).argmax(dim=1).item()
        orig_prob = torch.softmax(model(x_batch), dim=1).max().item()
        adv_pred = model(x_adv).argmax(dim=1).item()
        adv_prob = torch.softmax(model(x_adv), dim=1).max().item()
        delta = (x_adv - x_batch).abs().max().item()
    
    print(f"  Original: class={orig_pred}, prob={orig_prob:.4f}")
    print(f"  Adversarial: class={adv_pred}, prob={adv_prob:.4f}")
    print(f"  Max perturbation: {delta:.4f}")
    print(f"  Attack Success: {orig_pred != adv_pred}")
    
    print("\n[3] 测试 PGD 攻击...")
    pgd = PGD(model, device=device, eps=0.03, num_steps=40, random_start=True)
    x_adv = pgd.generate(x_batch, y_batch, targeted=False)
    
    with torch.no_grad():
        adv_pred = model(x_adv).argmax(dim=1).item()
        adv_prob = torch.softmax(model(x_adv), dim=1).max().item()
        delta = (x_adv - x_batch).abs().max().item()
    
    print(f"  Adversarial: class={adv_pred}, prob={adv_prob:.4f}")
    print(f"  Max perturbation: {delta:.4f}")
    print(f"  Attack Success: {orig_pred != adv_pred}")
    
    print("\n[4] 测试 SAP 攻击...")
    sap = SAP(model, device=device, eps=0.03, num_steps=40, use_pgd_init=False)
    x_adv = sap.generate(x_batch, y_batch, targeted=False)
    
    with torch.no_grad():
        adv_pred = model(x_adv).argmax(dim=1).item()
        adv_prob = torch.softmax(model(x_adv), dim=1).max().item()
        delta = (x_adv - x_batch).abs().max().item()
        smoothness = sap.smoothness_metric(x_adv - x_batch)
    
    print(f"  Adversarial: class={adv_pred}, prob={adv_prob:.4f}")
    print(f"  Max perturbation: {delta:.4f}")
    print(f"  Smoothness: {smoothness:.8f}")
    print(f"  Attack Success: {orig_pred != adv_pred}")

def compute_asr_debug(model, x_orig, x_adv, y_true):
    """
    正确的ASR计算：
    ASR = 原本预测正确的样本中，被攻击后预测错误的比例
    """
    with torch.no_grad():
        pred_orig = model(x_orig).argmax(dim=1)
        pred_adv = model(x_adv).argmax(dim=1)
        
        # 原本预测正确的样本
        originally_correct = (pred_orig == y_true)
        # 攻击后预测错误的样本
        now_incorrect = (pred_adv != y_true)
        
        # 攻击成功 = 原本正确 & 现在错误
        attack_success = originally_correct & now_incorrect
        
        if originally_correct.sum().item() == 0:
            return 0.0
            
        asr = attack_success.sum().item() / originally_correct.sum().item() * 100
        
        return asr, originally_correct.sum().item(), attack_success.sum().item()

def test_asr_computation():
    """测试ASR计算逻辑"""
    print("\n" + "="*60)
    print("测试 ASR 计算逻辑")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = ECG_CNN(num_classes=5).to(device)
    checkpoint = torch.load('checkpoints/clean_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试数据
    from torch.utils.data import DataLoader
    dataset = MITBIHDataset('data/mitbih_test.csv', transform='normalize', preload=True)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # 取一批数据
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    
    # FGSM攻击
    fgsm = FGSM(model, device=device, eps=0.03)
    x_adv = fgsm.generate(x, y, targeted=False)
    
    # 计算ASR
    asr, total_correct, success = compute_asr_debug(model, x, x_adv, y)
    
    print(f"\nFGSM Attack (eps=0.03):")
    print(f"  原本预测正确的样本数: {total_correct}")
    print(f"  攻击后预测错误的样本数: {success}")
    print(f"  ASR: {asr:.2f}%")
    
    # 检查原始准确率
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        acc = (pred == y).float().mean().item() * 100
    print(f"  原始准确率: {acc:.2f}%")

if __name__ == "__main__":
    debug_single_sample()
    test_asr_computation()
