"""
调试攻击 - 寻找模型不确定的样本进行测试
"""
import torch
import torch.nn.functional as F
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.ecg_cnn import ECG_CNN
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from data.mitbih_loader import MITBIHDataset

def find_uncertain_samples():
    """寻找模型不太确定的样本"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ECG_CNN(num_classes=5).to(device)
    checkpoint = torch.load('checkpoints/clean_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset = MITBIHDataset('data/mitbih_test.csv', transform='normalize', preload=True)
    
    print("寻找模型不太确定的样本（置信度<90%）...\n")
    
    uncertain_samples = []
    for i in range(1000):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
            max_prob = probs.max().item()
            pred = output.argmax().item()
        
        if pred == y and max_prob < 0.9 and max_prob > 0.6:  # 正确但不太确定
            uncertain_samples.append((i, y, pred, max_prob))
            if len(uncertain_samples) >= 5:
                break
    
    if not uncertain_samples:
        print("未找到不确定样本，模型整体置信度太高")
        return None
    
    print("找到的不确定样本:")
    for i, (idx, y, pred, prob) in enumerate(uncertain_samples):
        print(f"  {i+1}. Sample {idx}: True={y}, Pred={pred}, Prob={prob:.4f}")
    
    return uncertain_samples[0][0]  # 返回第一个样本的索引

def test_attack_on_sample(sample_idx, eps=0.03):
    """在特定样本上测试攻击"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ECG_CNN(num_classes=5).to(device)
    checkpoint = torch.load('checkpoints/clean_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset = MITBIHDataset('data/mitbih_test.csv', transform='normalize', preload=True)
    
    x, y = dataset[sample_idx]
    x = x.unsqueeze(0).to(device)
    y = torch.tensor([y]).to(device)
    
    with torch.no_grad():
        orig_output = model(x)
        orig_prob = torch.softmax(orig_output, dim=1)
        orig_pred = orig_output.argmax().item()
        orig_conf = orig_prob.max().item()
    
    print(f"\n样本 {sample_idx}:")
    print(f"  真实标签: {y.item()}, 预测: {orig_pred}, 置信度: {orig_conf:.4f}")
    print(f"  概率分布: {orig_prob.squeeze()}")
    
    # 测试FGSM
    print(f"\nFGSM攻击 (eps={eps}):")
    fgsm = FGSM(model, device=device, eps=eps)
    x_adv = fgsm.generate(x, y, targeted=False)
    
    with torch.no_grad():
        adv_output = model(x_adv)
        adv_prob = torch.softmax(adv_output, dim=1)
        adv_pred = adv_output.argmax().item()
        adv_conf = adv_prob.max().item()
        delta = (x_adv - x).abs().max().item()
    
    print(f"  预测: {adv_pred}, 置信度: {adv_conf:.4f}")
    print(f"  概率分布: {adv_prob.squeeze()}")
    print(f"  最大扰动: {delta:.4f}")
    print(f"  攻击成功: {orig_pred != adv_pred}")
    
    # 测试PGD
    print(f"\nPGD攻击 (eps={eps}, steps=40):")
    pgd = PGD(model, device=device, eps=eps, num_steps=40, random_start=True)
    x_adv = pgd.generate(x, y, targeted=False)
    
    with torch.no_grad():
        adv_output = model(x_adv)
        adv_pred = adv_output.argmax().item()
    
    print(f"  预测: {adv_pred}")
    print(f"  攻击成功: {orig_pred != adv_pred}")
    
    return orig_pred != adv_pred

def batch_attack_test(eps=0.03, num_samples=256):
    """批量测试攻击成功率"""
    from torch.utils.data import DataLoader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ECG_CNN(num_classes=5).to(device)
    checkpoint = torch.load('checkpoints/clean_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset = MITBIHDataset('data/mitbih_test.csv', transform='normalize', preload=True)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    
    # 只选模型预测正确的样本
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        correct_mask = (pred == y)
        x_correct = x[correct_mask]
        y_correct = y[correct_mask]
        pred_correct = pred[correct_mask]
    
    print(f"\n批量测试 ({eps=}):")
    print(f"  总样本: {len(y)}")
    print(f"  预测正确的样本: {len(y_correct)}")
    
    # FGSM攻击
    fgsm = FGSM(model, device=device, eps=eps)
    x_adv = fgsm.generate(x_correct, y_correct, targeted=False)
    
    with torch.no_grad():
        adv_pred = model(x_adv).argmax(dim=1)
        success = (adv_pred != y_correct).sum().item()
        asr = success / len(y_correct) * 100
    
    print(f"\nFGSM:")
    print(f"  攻击成功: {success}/{len(y_correct)}")
    print(f"  ASR: {asr:.2f}%")
    
    # PGD攻击
    pgd = PGD(model, device=device, eps=eps, num_steps=40, random_start=True)
    x_adv = pgd.generate(x_correct, y_correct, targeted=False)
    
    with torch.no_grad():
        adv_pred = model(x_adv).argmax(dim=1)
        success = (adv_pred != y_correct).sum().item()
        asr = success / len(y_correct) * 100
    
    print(f"\nPGD:")
    print(f"  攻击成功: {success}/{len(y_correct)}")
    print(f"  ASR: {asr:.2f}%")

if __name__ == "__main__":
    # 测试不同epsilon的批量攻击
    for eps in [0.01, 0.03, 0.05, 0.1]:
        batch_attack_test(eps=eps, num_samples=256)
        print("\n" + "="*60)
