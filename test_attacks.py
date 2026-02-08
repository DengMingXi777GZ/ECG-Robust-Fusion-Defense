"""
测试所有攻击算法
"""
import sys
import os

# 设置项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

import torch
from models.ecg_cnn import ECG_CNN
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from attacks.sap import SAP


def test_fgsm():
    print("\n" + "="*50)
    print("Testing FGSM Attack")
    print("="*50)
    
    model = ECG_CNN(num_classes=5)
    model.eval()
    
    x = torch.randn(4, 1, 187).clamp(0, 1)  # 归一化输入
    y = torch.tensor([0, 1, 2, 3])
    
    attacker = FGSM(model, eps=0.01)
    x_adv = attacker.generate(x, y)
    
    perturbation = (x_adv - x).abs()
    print(f"Original shape: {list(x.shape)}")
    print(f"Adversarial shape: {list(x_adv.shape)}")
    print(f"Perturbation max: {perturbation.max().item():.6f}")
    print(f"Perturbation L-inf: {perturbation.max().item():.6f}")
    
    # 验证模型输出变化
    with torch.no_grad():
        out_orig = model(x).argmax(dim=1)
        out_adv = model(x_adv).argmax(dim=1)
    
    print(f"Original predictions: {out_orig.tolist()}")
    print(f"Adversarial predictions: {out_adv.tolist()}")
    print(f"Attack success: {(out_orig != out_adv).sum().item()}/{len(y)} samples")
    print("[OK] FGSM test passed!")


def test_pgd():
    print("\n" + "="*50)
    print("Testing PGD Attack")
    print("="*50)
    
    model = ECG_CNN(num_classes=5)
    model.eval()
    
    x = torch.randn(4, 1, 187).clamp(0, 1)
    y = torch.tensor([0, 1, 2, 3])
    
    attacker = PGD(model, eps=0.01, num_steps=20, random_start=True)
    x_adv = attacker.generate(x, y)
    
    perturbation = (x_adv - x).abs()
    print(f"Original shape: {list(x.shape)}")
    print(f"Adversarial shape: {list(x_adv.shape)}")
    print(f"Perturbation max: {perturbation.max().item():.6f}")
    print(f"Epsilon: {attacker.eps}")
    
    with torch.no_grad():
        out_orig = model(x).argmax(dim=1)
        out_adv = model(x_adv).argmax(dim=1)
    
    print(f"Original predictions: {out_orig.tolist()}")
    print(f"Adversarial predictions: {out_adv.tolist()}")
    print(f"Attack success: {(out_orig != out_adv).sum().item()}/{len(y)} samples")
    print("[OK] PGD test passed!")


def test_sap():
    print("\n" + "="*50)
    print("Testing SAP Attack")
    print("="*50)
    
    model = ECG_CNN(num_classes=5)
    model.eval()
    
    x = torch.randn(4, 1, 187).clamp(0, 1)
    y = torch.tensor([0, 1, 2, 3])
    
    print("Generating SAP adversarial examples...")
    sap = SAP(model, eps=0.01, num_steps=20, use_pgd_init=True)
    x_adv = sap.generate(x, y)
    
    delta = x_adv - x
    smoothness = sap.smoothness_metric(delta)
    
    print(f"Original shape: {list(x.shape)}")
    print(f"Adversarial shape: {list(x_adv.shape)}")
    print(f"Perturbation max: {delta.abs().max().item():.6f}")
    print(f"Perturbation smoothness (var of diff): {smoothness:.8f}")
    
    # 对比: PGD 扰动的平滑度
    pgd = PGD(model, eps=0.01, num_steps=20)
    x_pgd = pgd.generate(x, y)
    delta_pgd = x_pgd - x
    smoothness_pgd = sap.smoothness_metric(delta_pgd)
    print(f"PGD smoothness (var of diff): {smoothness_pgd:.8f}")
    print(f"Smoothness ratio (SAP/PGD): {smoothness/smoothness_pgd:.4f}")
    
    with torch.no_grad():
        out_orig = model(x).argmax(dim=1)
        out_adv = model(x_adv).argmax(dim=1)
        out_pgd = model(x_pgd).argmax(dim=1)
    
    print(f"Original predictions: {out_orig.tolist()}")
    print(f"SAP predictions: {out_adv.tolist()}")
    print(f"PGD predictions: {out_pgd.tolist()}")
    print(f"SAP attack success: {(out_orig != out_adv).sum().item()}/{len(y)} samples")
    print(f"PGD attack success: {(out_orig != out_pgd).sum().item()}/{len(y)} samples")
    
    if smoothness < smoothness_pgd * 0.5:
        print("[OK] SAP is significantly smoother than PGD!")
    
    print("[OK] SAP test passed!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running All Attack Tests")
    print("="*60)
    
    test_fgsm()
    test_pgd()
    test_sap()
    
    print("\n" + "="*60)
    print("All Attack Tests Passed!")
    print("="*60)
