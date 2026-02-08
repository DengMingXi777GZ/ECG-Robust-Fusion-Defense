"""
融合模型评估脚本 (Fusion Model Evaluation)
Layer 3 特征融合层 - 生成论文 Table 3 的对比实验结果

评估目标:
    - 对比 Clean Model、Standard AT、Fusion、Fusion+Detection 的性能
    - 在 Clean、PGD-20、SAP 攻击下评估
    - 统计各模型的参数量

输出:
    - results/model_comparison.csv: 对比表格
    - results/model_comparison.json: 详细结果
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from tabulate import tabulate

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ecg_cnn import ECG_CNN
from models.fusion_model import DualBranchECG
from models.adversarial_detector import AdversarialDetector
from attacks import PGD, SAP
from data.mitbih_loader import MITBIHDataset
from torch.utils.data import DataLoader


class ModelWrapper(nn.Module):
    """
    融合模型包装器，使双输入模型兼容攻击接口
    
    攻击层只传递一个参数 (x)，需要从预提取的特征中获取手工特征
    """
    
    def __init__(self, fusion_model, handcrafted_features, device='cpu'):
        """
        Args:
            fusion_model: DualBranchECG 模型
            handcrafted_features: [N, 12] 预提取的手工特征
            device: 计算设备
        """
        super().__init__()
        self.model = fusion_model
        self.handcrafted = torch.from_numpy(handcrafted_features).float().to(device)
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, 1, 187] ECG 信号
        
        Returns:
            logits: [B, num_classes] 分类 logits
        """
        batch_size = x.size(0)
        
        # 获取当前批次对应的手工特征
        # 假设输入批次是连续的，这在测试集按顺序评估时成立
        # 对于非顺序访问，需要通过其他方式索引
        
        # 这里我们假设 x 来自测试集，使用索引映射
        # 实际实现中，我们通过全局索引跟踪
        if hasattr(self, '_current_indices'):
            hc = self.handcrafted[self._current_indices]
        else:
            # 默认使用对应位置的特征（按顺序评估）
            # 通过 batch_counter 跟踪位置
            if not hasattr(self, '_batch_counter'):
                self._batch_counter = 0
            start_idx = self._batch_counter * batch_size
            end_idx = min(start_idx + batch_size, len(self.handcrafted))
            hc = self.handcrafted[start_idx:end_idx]
            self._batch_counter += 1
        
        # 确保特征批次大小匹配
        if hc.size(0) < batch_size:
            # 处理最后一个不完整的批次
            hc = self.handcrafted[-batch_size:]
        
        # 前向传播（只返回 logits，不返回特征）
        logits, _, _ = self.model(x, hc[:batch_size])
        return logits
    
    def reset_batch_counter(self):
        """重置批次计数器"""
        self._batch_counter = 0
    
    def set_indices(self, indices):
        """设置当前批次的索引"""
        self._current_indices = indices


class FusionWithDetector(nn.Module):
    """
    带检测器的融合模型
    
    先用检测器判断是否为对抗样本，如果是则拒绝预测，
    否则用融合模型进行分类。
    """
    
    def __init__(self, fusion_model, detector, handcrafted_features, 
                 device='cpu', reject_threshold=0.5):
        """
        Args:
            fusion_model: DualBranchECG 模型
            detector: AdversarialDetector 检测器
            handcrafted_features: [N, 12] 预提取的手工特征
            device: 计算设备
            reject_threshold: 拒绝阈值，检测概率大于此值则拒绝
        """
        super().__init__()
        self.fusion_model = fusion_model
        self.detector = detector
        self.handcrafted = torch.from_numpy(handcrafted_features).float().to(device)
        self.device = device
        self.reject_threshold = reject_threshold
        
        self.fusion_model.to(device)
        self.detector.to(device)
        self.fusion_model.eval()
        self.detector.eval()
        
        # 跟踪指标
        self.total_samples = 0
        self.rejected_samples = 0
    
    def forward(self, x):
        """
        前向传播（带检测）
        
        Args:
            x: [B, 1, 187] ECG 信号
        
        Returns:
            logits: [B, num_classes] 分类 logits
                    被拒绝的样本返回全零 logits（表示不确定）
        """
        batch_size = x.size(0)
        
        # 获取手工特征
        if hasattr(self, '_current_indices'):
            hc = self.handcrafted[self._current_indices]
        else:
            if not hasattr(self, '_batch_counter'):
                self._batch_counter = 0
            start_idx = self._batch_counter * batch_size
            end_idx = min(start_idx + batch_size, len(self.handcrafted))
            hc = self.handcrafted[start_idx:end_idx]
            self._batch_counter += 1
        
        if hc.size(0) < batch_size:
            hc = self.handcrafted[-batch_size:]
        hc = hc[:batch_size]
        
        # 检测对抗样本
        with torch.no_grad():
            detection_prob, _, _, _ = self.detector(x, hc)
            is_adversarial = (detection_prob > self.reject_threshold).squeeze()
        
        # 分类
        logits, _, _ = self.fusion_model(x, hc)
        
        # 标记被拒绝的样本（将 logits 置为 -inf 表示拒绝）
        # 实际评估时，被拒绝的样本不计入正确率
        if is_adversarial.any():
            logits[is_adversarial] = float('-inf')
        
        self.total_samples += batch_size
        self.rejected_samples += is_adversarial.sum().item()
        
        return logits
    
    def reset_batch_counter(self):
        """重置批次计数器"""
        self._batch_counter = 0
        self.total_samples = 0
        self.rejected_samples = 0
    
    def set_indices(self, indices):
        """设置当前批次的索引"""
        self._current_indices = indices
    
    def get_rejection_rate(self):
        """获取拒绝率"""
        if self.total_samples == 0:
            return 0.0
        return self.rejected_samples / self.total_samples


def load_handcrafted_features(data_dir='data'):
    """
    加载预计算的手工特征
    
    Args:
        data_dir: 数据目录
    
    Returns:
        test_features: [N, 12] 测试集手工特征
    """
    features_path = os.path.join(data_dir, 'handcrafted_features_test.npy')
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"手工特征文件不存在: {features_path}\n"
            f"请先运行: python features/extract_mitbih_features.py"
        )
    
    features = np.load(features_path)
    print(f"[OK] 加载手工特征: {features.shape}")
    return features


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def evaluate_model(model, test_loader, device='cpu', model_name='Model'):
    """
    评估模型在干净样本上的准确率
    
    Args:
        model: 待评估模型
        test_loader: 测试数据加载器
        device: 计算设备
        model_name: 模型名称
    
    Returns:
        accuracy: 准确率 (%)
    """
    model.eval()
    correct = 0
    total = 0
    
    # 重置批次计数器（用于融合模型）
    if hasattr(model, 'reset_batch_counter'):
        model.reset_batch_counter()
    
    pbar = tqdm(test_loader, desc=f"Eval {model_name} (Clean)")
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            
            # 处理被拒绝的样本（logits 为 -inf）
            valid_mask = ~(output == float('-inf')).all(dim=1)
            if valid_mask.any():
                pred = output[valid_mask].argmax(dim=1)
                correct += (pred == y[valid_mask]).sum().item()
                total += valid_mask.sum().item()
            
            pbar.set_postfix({'acc': f'{100.*correct/max(total,1):.2f}%'})
    
    accuracy = 100.0 * correct / max(total, 1)
    return accuracy


def evaluate_robustness(model, test_loader, handcrafted_features, 
                        eps=0.05, device='cpu', model_name='Model'):
    """
    评估模型在对抗攻击下的鲁棒性
    
    Args:
        model: 待评估模型（需支持单输入接口）
        test_loader: 测试数据加载器
        handcrafted_features: [N, 12] 手工特征
        eps: 扰动预算
        device: 计算设备
        model_name: 模型名称
    
    Returns:
        results: 字典，包含 PGD-20 和 SAP 的准确率
    """
    model.eval()
    
    # 创建攻击器
    attacks = {
        'PGD-20': PGD(model, device=device, eps=eps, num_steps=20, alpha=eps/5),
        'SAP': SAP(model, device=device, eps=eps, num_steps=40, lr=0.01)
    }
    
    results = {}
    
    for attack_name, attacker in attacks.items():
        correct = 0
        total = 0
        idx = 0
        batch_size = test_loader.batch_size
        
        # 重置批次计数器
        if hasattr(model, 'reset_batch_counter'):
            model.reset_batch_counter()
        
        pbar = tqdm(test_loader, desc=f"Eval {model_name} ({attack_name})")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            # 设置当前批次的索引
            if hasattr(model, 'set_indices'):
                indices = torch.arange(idx, idx + x.size(0))
                model.set_indices(indices)
            
            # 生成对抗样本
            x_adv = attacker.generate(x, y)
            
            # 评估
            with torch.no_grad():
                output = model(x_adv)
                
                # 处理被拒绝的样本
                valid_mask = ~(output == float('-inf')).all(dim=1)
                if valid_mask.any():
                    pred = output[valid_mask].argmax(dim=1)
                    correct += (pred == y[valid_mask]).sum().item()
                    total += valid_mask.sum().item()
            
            idx += x.size(0)
            pbar.set_postfix({'acc': f'{100.*correct/max(total,1):.2f}%'})
        
        accuracy = 100.0 * correct / max(total, 1)
        results[attack_name] = accuracy
        
        # 对于带检测器的模型，打印拒绝率
        if hasattr(model, 'get_rejection_rate'):
            rejection_rate = model.get_rejection_rate()
            print(f"  [{model_name}] {attack_name} Rejection Rate: {rejection_rate*100:.2f}%")
    
    return results


def load_model_with_params(model_path, device='cpu'):
    """
    加载模型并计算参数量
    
    Args:
        model_path: 模型路径
        device: 计算设备
    
    Returns:
        model: 加载的模型
        num_params: 参数量
    """
    model = ECG_CNN(num_classes=5).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    num_params = count_parameters(model)
    return model, num_params


def evaluate_all_models(test_loader, handcrafted_features, eps=0.05, device='cpu'):
    """
    评估所有模型
    
    Args:
        test_loader: 测试数据加载器
        handcrafted_features: [N, 12] 手工特征
        eps: 扰动预算
        device: 计算设备
    
    Returns:
        results: 评估结果字典
    """
    results = {}
    
    # 1. Clean Model (Layer 1 基线)
    print("\n" + "="*60)
    print("[1/5] 评估 Clean Model (Layer 1 基线)")
    print("="*60)
    
    clean_model_path = 'checkpoints/clean_model.pth'
    if os.path.exists(clean_model_path):
        model, num_params = load_model_with_params(clean_model_path, device)
        clean_acc = evaluate_model(model, test_loader, device, 'Clean')
        robustness = evaluate_robustness(model, test_loader, handcrafted_features, eps, device, 'Clean')
        
        results['Clean (Layer 1)'] = {
            'Clean': clean_acc,
            'PGD-20': robustness['PGD-20'],
            'SAP': robustness['SAP'],
            'Params': num_params / 1000  # 转换为 K
        }
        print(f"Results: Clean={clean_acc:.2f}%, PGD-20={robustness['PGD-20']:.2f}%, SAP={robustness['SAP']:.2f}%")
    else:
        print(f"⚠️  模型不存在: {clean_model_path}")
    
    # 2. Standard AT
    print("\n" + "="*60)
    print("[2/5] 评估 Standard AT")
    print("="*60)
    
    at_model_path = 'checkpoints/adv_standard_at.pth'
    if os.path.exists(at_model_path):
        model, num_params = load_model_with_params(at_model_path, device)
        clean_acc = evaluate_model(model, test_loader, device, 'Standard AT')
        robustness = evaluate_robustness(model, test_loader, handcrafted_features, eps, device, 'Standard AT')
        
        results['Standard AT'] = {
            'Clean': clean_acc,
            'PGD-20': robustness['PGD-20'],
            'SAP': robustness['SAP'],
            'Params': num_params / 1000
        }
        print(f"Results: Clean={clean_acc:.2f}%, PGD-20={robustness['PGD-20']:.2f}%, SAP={robustness['SAP']:.2f}%")
    else:
        print(f"⚠️  模型不存在: {at_model_path}")
    
    # 3. AT+NSR (最佳 Layer 2 模型)
    print("\n" + "="*60)
    print("[3/5] 评估 AT+NSR (最佳 Layer 2 模型)")
    print("="*60)
    
    nsr_model_path = 'checkpoints/at_nsr.pth'
    if os.path.exists(nsr_model_path):
        model, num_params = load_model_with_params(nsr_model_path, device)
        clean_acc = evaluate_model(model, test_loader, device, 'AT+NSR')
        robustness = evaluate_robustness(model, test_loader, handcrafted_features, eps, device, 'AT+NSR')
        
        results['AT+NSR'] = {
            'Clean': clean_acc,
            'PGD-20': robustness['PGD-20'],
            'SAP': robustness['SAP'],
            'Params': num_params / 1000
        }
        print(f"Results: Clean={clean_acc:.2f}%, PGD-20={robustness['PGD-20']:.2f}%, SAP={robustness['SAP']:.2f}%")
    else:
        print(f"⚠️  模型不存在: {nsr_model_path}")
    
    # 4. Fusion Model
    print("\n" + "="*60)
    print("[4/5] 评估 Fusion Model")
    print("="*60)
    
    fusion_model_path = 'checkpoints/fusion_best.pth'
    if os.path.exists(fusion_model_path):
        # 加载融合模型
        fusion_model = DualBranchECG(num_classes=5, pretrained_path=None, freeze_deep_branch=False)
        checkpoint = torch.load(fusion_model_path, map_location=device, weights_only=False)
        fusion_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        fusion_model.to(device)
        fusion_model.eval()
        
        # 包装模型以兼容攻击接口
        wrapped_model = ModelWrapper(fusion_model, handcrafted_features, device)
        
        num_params = count_parameters(fusion_model)
        clean_acc = evaluate_model(wrapped_model, test_loader, device, 'Fusion')
        robustness = evaluate_robustness(wrapped_model, test_loader, handcrafted_features, eps, device, 'Fusion')
        
        results['Fusion (Ours)'] = {
            'Clean': clean_acc,
            'PGD-20': robustness['PGD-20'],
            'SAP': robustness['SAP'],
            'Params': num_params / 1000
        }
        print(f"Results: Clean={clean_acc:.2f}%, PGD-20={robustness['PGD-20']:.2f}%, SAP={robustness['SAP']:.2f}%")
    else:
        print(f"⚠️  融合模型不存在: {fusion_model_path}")
        print("提示: 请先运行 train_fusion.py 训练融合模型")
    
    # 5. Fusion + Detection
    print("\n" + "="*60)
    print("[5/5] 评估 Fusion + Detection")
    print("="*60)
    
    detector_model_path = 'checkpoints/detector_best.pth'
    if os.path.exists(fusion_model_path) and os.path.exists(detector_model_path):
        # 加载融合模型
        fusion_model = DualBranchECG(num_classes=5, pretrained_path=None, freeze_deep_branch=False)
        checkpoint = torch.load(fusion_model_path, map_location=device, weights_only=False)
        fusion_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        fusion_model.to(device)
        fusion_model.eval()
        
        # 加载检测器
        detector = AdversarialDetector(fusion_model)
        checkpoint = torch.load(detector_model_path, map_location=device, weights_only=False)
        detector.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        detector.to(device)
        detector.eval()
        
        # 包装模型
        wrapped_model = FusionWithDetector(fusion_model, detector, handcrafted_features, device)
        
        num_params = count_parameters(fusion_model) + count_parameters(detector)
        clean_acc = evaluate_model(wrapped_model, test_loader, device, 'Fusion+Detection')
        robustness = evaluate_robustness(wrapped_model, test_loader, handcrafted_features, eps, device, 'Fusion+Detection')
        
        results['Fusion+Detection'] = {
            'Clean': clean_acc,
            'PGD-20': robustness['PGD-20'],
            'SAP': robustness['SAP'],
            'Params': num_params / 1000
        }
        print(f"Results: Clean={clean_acc:.2f}%, PGD-20={robustness['PGD-20']:.2f}%, SAP={robustness['SAP']:.2f}%")
    else:
        if not os.path.exists(fusion_model_path):
            print(f"⚠️  融合模型不存在: {fusion_model_path}")
        if not os.path.exists(detector_model_path):
            print(f"⚠️  检测器模型不存在: {detector_model_path}")
        print("提示: 请先训练融合模型和检测器")
    
    return results


def generate_comparison_table(results, output_dir='results'):
    """
    生成对比表格
    
    Args:
        results: 评估结果字典
        output_dir: 输出目录
    
    Returns:
        df: 对比表格 DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建 DataFrame
    df = pd.DataFrame(results).T
    
    # 确保列顺序正确
    columns = ['Clean', 'PGD-20', 'SAP', 'Params']
    df = df[[col for col in columns if col in df.columns]]
    
    # 格式化参数量
    if 'Params' in df.columns:
        df['Params'] = df['Params'].apply(lambda x: f"{x:.1f}K")
    
    # 保存为 CSV
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(csv_path)
    print(f"\n[OK] 对比表格已保存: {csv_path}")
    
    # 保存为 JSON
    json_path = os.path.join(output_dir, 'model_comparison.json')
    with open(json_path, 'w') as f:
        # 将结果转换为可序列化的格式
        json_results = {}
        for model_name, metrics in results.items():
            json_results[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
        json.dump(json_results, f, indent=2)
    print(f"[OK] 详细结果已保存: {json_path}")
    
    return df


def print_comparison_table(df):
    """
    打印对比表格
    
    Args:
        df: 对比表格 DataFrame
    """
    print("\n" + "="*70)
    print("模型对比评估结果")
    print("="*70)
    
    # 准备打印数据
    print_data = []
    for model_name, row in df.iterrows():
        print_data.append([
            model_name,
            f"{row['Clean']:.1f}%",
            f"{row['PGD-20']:.1f}%",
            f"{row['SAP']:.1f}%",
            row['Params']
        ])
    
    headers = ['模型', 'Clean', 'PGD-20', 'SAP', '参数量']
    print(tabulate(print_data, headers=headers, tablefmt='grid'))
    print("="*70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Fusion Model Evaluation')
    
    parser.add_argument('--eps', type=float, default=0.05, 
                        help='Epsilon for adversarial attacks (default: 0.05)')
    parser.add_argument('--batch-size', type=int, default=256, 
                        help='Batch size (default: 256)')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device (cuda/cpu, default: cuda)')
    parser.add_argument('--output-dir', type=str, default='results', 
                        help='Output directory (default: results)')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式：只在部分数据上评估')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载手工特征
    print("\n加载手工特征...")
    try:
        handcrafted_features = load_handcrafted_features()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_dataset = MITBIHDataset('data/mitbih_test.csv', transform='normalize', preload=True)
    
    # 快速模式：只使用部分数据
    if args.quick:
        print("[Quick Mode] 只评估前 1000 个样本")
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, range(min(1000, len(test_dataset))))
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 评估所有模型
    print("\n开始评估...")
    results = evaluate_all_models(test_loader, handcrafted_features, args.eps, device)
    
    if not results:
        print("\n⚠️ 没有可评估的模型，请先训练模型")
        return
    
    # 生成对比表格
    df = generate_comparison_table(results, args.output_dir)
    
    # 打印对比表格
    print_comparison_table(df)
    
    print("\n[OK] 评估完成!")


if __name__ == "__main__":
    main()
