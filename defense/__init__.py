"""
防御层模块 (Defense Layer)
Layer 2: 对抗训练 + NSR正则化

模块列表:
    - adv_trainer: 对抗训练数据集生成器
    - nsr_loss: NSR正则化损失函数
    - train_standard_at: 标准对抗训练 (Madry's AT)
    - train_nsr: NSR正则化训练
    - train_at_nsr: 联合训练 (AT + NSR)

使用方法:
    # 标准对抗训练
    from defense.train_standard_at import train_standard_at
    model, history = train_standard_at(eps=0.05, epochs=50)
    
    # NSR训练
    from defense.train_nsr import train_nsr
    model, history = train_nsr(beta=0.4, eps=0.05, epochs=50)
    
    # 联合训练
    from defense.train_at_nsr import train_combined
    model, history = train_combined(beta=0.4, eps=0.05, epochs=50)
"""

from .adv_trainer import AdversarialDataset, MixedAdversarialDataset
from .nsr_loss import NSRLoss, NSRLossV2

__all__ = [
    'AdversarialDataset',
    'MixedAdversarialDataset', 
    'NSRLoss',
    'NSRLossV2'
]
