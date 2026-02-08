"""
对抗攻击模块
"""
from .base_attack import BaseAttack
from .fgsm import FGSM
from .pgd import PGD
from .sap import SAP

__all__ = ['BaseAttack', 'FGSM', 'PGD', 'SAP']
