Deep learning pipeline for classifying morphological changes in kidney glomeruli from PAS-stained histopathological images. This repository contains the code accompanying the doctoral dissertation:
> **Optimierungsstrategien für die Deep-Learning-basierte Klassifikation morphologischer Veränderungen in Nierenglomeruli**  


## Overview

This project systematically evaluates five neural network architectures and multiple training strategies for the classification of nine glomerular morphological patterns:

| # | Category | Description |
|---|----------|-------------|
| 1 | Unauffälliger Glomerulus | Normal glomerulus |
| 2 | Amyloidose | Amyloidosis |
| 3 | Noduläre Sklerose | Nodular sclerosis (diabetic) |
| 4 | Globale Sklerose | Global sclerosis |
| 5 | Mesangiale Matrixvermehrung | Mesangial matrix expansion |
| 6 | MPGN | Membranoproliferative GN |
| 7 | Nekrose | Necrosis |
| 8 | Segmentale Sklerose | Segmental sclerosis |
| 9 | Andere Strukturen | Other structures |

### Models

| Model | Parameters | Pretraining |
|-------|-----------|-------------|
| ResNet-50 | 23.5M | ImageNet-1k |
| ConvNeXt-Tiny | 27.8M | ImageNet-1k |
| SwinV2-Tiny | 27.6M | ImageNet-1k |
| ViT-L/16 | 303.3M | ImageNet-21k |
| Phikon-v2 | 303.4M | Pan-Cancer histopathology (DINOv2 SSL) |

### Training Strategies

- **Baseline**: Standard fine-tuning with Macenko stain normalization
- **RandAugment + Label Smoothing**: Data augmentation and regularization
- **Weighted Random Sampler (WRS)**: Sampling-based class imbalance compensation
- **Weighted Cross-Entropy (WCE)**: Loss-based class reweighting (ENS)
- **LDAM + DRW**: Label-Distribution-Aware Margin Loss with Deferred Re-Weighting

## Acknowledgments

- Pathlogisches Institut, Universitätsmedizin Mannheim for providing the histopathological dataset
- Phikon-v2 model by Filiot et al. (2024)
- LDAM implementation based on Cao et al. (2019)
