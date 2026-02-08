"""
1D-CNN ECG 分类模型
架构参考: Ma & Liang 2022
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECG_CNN(nn.Module):
    """
    1D-CNN 用于 ECG 五分类 (N, S, V, F, Q)
    
    架构:
        - 4个 Conv Blocks: Conv1d -> BN -> ReLU -> MaxPool
        - Global Average Pooling
        - 2个 FC layers
    """
    
    def __init__(self, num_classes=5, input_channels=1, input_length=187):
        super(ECG_CNN, self).__init__()
        
        # Block 1: in=1, out=16, kernel=7, stride=1, padding=3
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 187 -> 93
        
        # Block 2: in=16, out=32, kernel=5, stride=1, padding=2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 93 -> 46
        
        # Block 3: in=32, out=64, kernel=3, stride=1, padding=1
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 46 -> 23
        
        # Block 4: in=64, out=128, kernel=3, stride=1, padding=1
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # 23 -> 11
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, 1, 187]
        Returns:
            logits: [batch_size, num_classes]
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Global Average Pooling
        x = self.global_pool(x)  # [B, 128, 1]
        x = x.view(x.size(0), -1)  # [B, 128]
        
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """测试模型"""
    model = ECG_CNN(num_classes=5)
    
    # 测试输入
    batch_size = 8
    x = torch.randn(batch_size, 1, 187)
    
    # 前向传播
    output = model(x)
    
    print(f"Model parameter count: {model.count_parameters():,}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 验证参数量 < 500K
    assert model.count_parameters() < 500000, "Model too large!"
    assert output.shape == (batch_size, 5), "Output shape mismatch!"
    
    print("\n[OK] Model test passed!")
    
    return model


if __name__ == "__main__":
    test_model()
