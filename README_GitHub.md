# ECG Adversarial Defense System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive three-layer defense system against adversarial attacks on ECG deep learning models.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Attack Layer                                       │
│  ├── FGSM, PGD attacks                                       │
│  └── SAP (Smooth Adversarial Perturbation)                   │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Defense Layer                                      │
│  ├── Standard Adversarial Training                           │
│  └── NSR (Noise-to-Signal Ratio) Regularization              │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Feature Fusion Layer ⭐                            │
│  ├── 12-dim handcrafted features (RR, QRS, etc.)            │
│  └── Dual-branch fusion network + Adversarial Detector      │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Results

| Model | Clean | PGD-20 | SAP | Params |
|-------|-------|--------|-----|--------|
| Baseline | 93.43% | 15.01% | 84.79% | 42K |
| Standard AT | 96.04% | 92.08% | 93.85% | 42K |
| AT+NSR | 95.28% | 91.53% | 93.40% | 42K |
| **Fusion (Ours)** | **95.22%** | **93.20%** | **94.60%** | **55K** |

**Adversarial Detector**: AUC-ROC = 0.9217

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ECG-Adversarial-Defense.git
cd ECG-Adversarial-Defense

# Install dependencies
pip install -r requirements.txt
```

### Train Baseline Model

```bash
python train_baseline.py
```

### Generate Adversarial Examples

```bash
python generate_adversarial_dataset.py --eps 0.05
```

### Train Defense Model

```bash
python defense/train_at_nsr.py --eps 0.05 --beta 0.4
```

### Train Fusion Model (Layer 3)

```bash
# Extract handcrafted features
python features/extract_mitbih_features.py
python features/extract_train_features.py

# Train fusion model
python train_fusion.py --epochs_stage1 10 --epochs_stage2 5

# Train adversarial detector
python -c "from models.adversarial_detector import main; main()"
```

## 📁 Project Structure

```
├── attacks/              # Attack algorithms (FGSM, PGD, SAP)
├── defense/              # Defense training (AT, NSR)
├── models/               # Model definitions
│   ├── ecg_cnn.py        # Baseline 1D-CNN
│   ├── fusion_model.py   # Dual-branch fusion ⭐
│   └── adversarial_detector.py
├── features/             # Handcrafted feature extraction ⭐
├── data/                 # Data loaders
├── evaluation/           # Evaluation scripts
├── visualization/        # Visualization tools
└── checkpoints/          # Model weights
```

## 📚 Key Features

### 1. SAP Attack (Nature Medicine 2020)
- Smooth adversarial perturbation using multi-scale Gaussian kernels
- 85% lower smoothness than PGD

### 2. NSR Regularization
- Local Lipschitz constraint via gradient regularization
- Improves ACC_robust from 0.66 to 0.94

### 3. Feature Fusion Defense ⭐
- 12-dim physiological features: RR intervals, QRS width, etc.
- Feature invariance analysis shows RR features are robust to attacks
- Dual-branch architecture: Deep CNN + Handcrafted MLP

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{han2020deep,
  title={Deep learning models for electrocardiograms are susceptible to adversarial attack},
  author={Han, Chengzong and Ribeiro, Ant{\^o}nio H and Haimi-Cohen, Raphael and Khera, Rohan and de Oliveira, Filipe M and Balaji, Aarthi and Ramchand, Jaideep and Krishna, Vinodkumar and Rajpurkar, Pranav},
  journal={Nature Medicine},
  year={2020}
}

@inproceedings{ma2022explainable,
  title={Explainable deep learning for efficient adversarial defense},
  author={Ma, Xingjun and Liang, Ningning},
  booktitle={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022}
}
```

## 📧 Contact

For questions or suggestions, please open an issue or contact the author.

## License

This project is licensed under the MIT License.
