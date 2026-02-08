"""
双分支融合网络模型 (Dual-Branch Fusion Model)
Deep CNN Branch + Handcrafted Features Branch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.ecg_cnn import ECG_CNN
except ImportError:
    from ecg_cnn import ECG_CNN


class DualBranchECG(nn.Module):
    """
    双分支 ECG 分类网络
    
    架构:
        - Deep Branch: 预训练的 CNN (来自 Layer 2)
        - Handcrafted Branch: 轻量级 MLP 处理 12 维手工特征
        - Fusion Layer: 特征拼接 + 分类
    """
    
    def __init__(self, num_classes=5, pretrained_path='checkpoints/at_nsr.pth', 
                 freeze_deep_branch=False):
        super(DualBranchECG, self).__init__()
        
        self.num_classes = num_classes
        self.deep_feature_dim = 128
        self.handcrafted_dim = 12
        self.hc_hidden_dim = 32
        self.hc_output_dim = 16
        
        # ==================== Deep Branch ====================
        self.deep_branch = ECG_CNN(num_classes=num_classes)
        
        # 加载预训练权重
        if pretrained_path and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        try:
            checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
            self.deep_branch.load_state_dict(checkpoint['model_state_dict'])
            print(f"[OK] 加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"[Warning] 无法加载预训练权重: {e}")
            print("[Warning] 使用随机初始化的 Deep Branch")
        
        # 修改 Deep Branch 的最后一层，输出 128 维特征
        # 原架构: FC(128, 64) -> ReLU -> Dropout -> FC(64, 5)
        # 新架构: FC(128, 128) -> ReLU (特征提取层)
        self.deep_branch.fc1 = nn.Linear(128, self.deep_feature_dim)
        self.deep_branch.fc2 = nn.Identity()  # 移除原来的分类层
        
        # 冻结 Deep Branch (可选)
        if freeze_deep_branch:
            for param in self.deep_branch.parameters():
                param.requires_grad = False
            print("[Info] Deep Branch 已冻结")
        
        # ==================== Handcrafted Branch ====================
        self.handcrafted_branch = nn.Sequential(
            nn.Linear(self.handcrafted_dim, self.hc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hc_hidden_dim, self.hc_output_dim),
            nn.ReLU()
        )
        
        # ==================== Fusion Layer ====================
        fusion_input_dim = self.deep_feature_dim + self.hc_output_dim  # 128 + 16 = 144
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
        
        # 计算参数量
        total_params = self.count_parameters()
        print(f"[Info] 融合模型总参数量: {total_params:,} ({total_params/1000:.1f}K)")
    
    def _initialize_weights(self):
        """初始化融合层的权重"""
        for m in self.handcrafted_branch.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x_signal, x_handcrafted):
        """
        前向传播
        
        Args:
            x_signal: [B, 1, 187] 原始 ECG 信号
            x_handcrafted: [B, 12] 手工特征
        
        Returns:
            output: [B, num_classes] 分类 logits
            deep_feat: [B, 128] 深度特征
            hc_feat: [B, 16] 手工特征
        """
        # Deep features
        deep_feat = self.deep_branch(x_signal)  # [B, 128]
        
        # Handcrafted features
        hc_feat = self.handcrafted_branch(x_handcrafted)  # [B, 16]
        
        # Fusion
        combined = torch.cat([deep_feat, hc_feat], dim=1)  # [B, 144]
        output = self.fusion(combined)  # [B, num_classes]
        
        return output, deep_feat, hc_feat
    
    def get_deep_features(self, x_signal):
        """仅获取深度特征（用于特征分析）"""
        return self.deep_branch(x_signal)
    
    def get_handcrafted_features(self, x_handcrafted):
        """仅获取手工特征（用于特征分析）"""
        return self.handcrafted_branch(x_handcrafted)
    
    def count_parameters(self):
        """计算可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_deep_branch(self):
        """冻结 Deep Branch"""
        for param in self.deep_branch.parameters():
            param.requires_grad = False
        print("[Info] Deep Branch 已冻结")
    
    def unfreeze_deep_branch(self):
        """解冻 Deep Branch"""
        for param in self.deep_branch.parameters():
            param.requires_grad = True
        print("[Info] Deep Branch 已解冻")


class DeepOnlyModel(nn.Module):
    """
    仅使用 Deep Branch 的模型（用于对比实验）
    """
    
    def __init__(self, num_classes=5, pretrained_path='checkpoints/at_nsr.pth'):
        super(DeepOnlyModel, self).__init__()
        
        self.deep_branch = ECG_CNN(num_classes=num_classes)
        
        # 加载预训练权重
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            self.deep_branch.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"[Warning] 无法加载预训练权重: {e}")
    
    def forward(self, x_signal, x_handcrafted=None):
        """前向传播，忽略手工特征"""
        return self.deep_branch(x_signal)


class HandcraftedOnlyModel(nn.Module):
    """
    仅使用 Handcrafted Branch 的模型（用于对比实验）
    """
    
    def __init__(self, num_classes=5):
        super(HandcraftedOnlyModel, self).__init__()
        
        self.handcrafted_branch = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x_signal=None, x_handcrafted=None):
        """前向传播，忽略原始信号"""
        hc_feat = self.handcrafted_branch(x_handcrafted)
        output = self.classifier(hc_feat)
        return output


def test_fusion_model():
    """测试融合模型"""
    print("=" * 60)
    print("双分支融合模型测试")
    print("=" * 60)
    
    # 检查是否有预训练权重
    pretrained_path = 'checkpoints/at_nsr.pth'
    if not torch.cuda.is_available():
        print("[Info] 使用 CPU 模式")
    
    # 创建模型
    model = DualBranchECG(
        num_classes=5,
        pretrained_path=pretrained_path,
        freeze_deep_branch=False
    )
    
    # 测试输入
    batch_size = 8
    x_signal = torch.randn(batch_size, 1, 187)
    x_handcrafted = torch.randn(batch_size, 12)
    
    # 前向传播
    output, deep_feat, hc_feat = model(x_signal, x_handcrafted)
    
    print(f"\n输入:")
    print(f"  x_signal: {x_signal.shape}")
    print(f"  x_handcrafted: {x_handcrafted.shape}")
    
    print(f"\n输出:")
    print(f"  logits: {output.shape}")
    print(f"  deep_feat: {deep_feat.shape}")
    print(f"  hc_feat: {hc_feat.shape}")
    
    print(f"\n参数量:")
    print(f"  总参数量: {model.count_parameters():,}")
    print(f"  Deep Branch: {sum(p.numel() for p in model.deep_branch.parameters()):,}")
    print(f"  Handcrafted Branch: {sum(p.numel() for p in model.handcrafted_branch.parameters()):,}")
    print(f"  Fusion Layer: {sum(p.numel() for p in model.fusion.parameters()):,}")
    
    # 验证输出形状
    assert output.shape == (batch_size, 5), f"输出形状错误: {output.shape}"
    assert deep_feat.shape == (batch_size, 128), f"Deep 特征形状错误: {deep_feat.shape}"
    assert hc_feat.shape == (batch_size, 16), f"Handcrafted 特征形状错误: {hc_feat.shape}"
    
    # 验证参数量 < 100K
    assert model.count_parameters() < 100000, f"模型太大: {model.count_parameters()}"
    
    print("\n" + "=" * 60)
    print("[OK] 融合模型测试通过!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    test_fusion_model()
