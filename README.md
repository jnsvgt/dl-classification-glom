# Glomerular Morphology Classification

Deep learning pipeline for classifying morphological patterns in PAS-stained kidney biopsy images. Developed as part of a doctoral thesis on computational nephropathology.

## Morphological Patterns

| # | Pattern | Description |
|---|---------|-------------|
| 1 | Unauffälliger Glomerulus | Normal glomerulus |
| 2 | Amyloidose | Amyloidosis |
| 3 | DM nodulär | Diabetic nephropathy (nodular) |
| 4 | Globale Vernarbung | Global sclerosis |
| 5 | Mesangiale Matrixvermehrung | Mesangial matrix expansion |
| 6 | MPGN | Membranoproliferative GN |
| 7 | Nekrose | Necrosis |
| 8 | Segmentale Vernarbung | Segmental sclerosis |
| 9 | Andere Strukturen | Other structures |

## Models

- **ResNet-50** — torchvision pretrained (ImageNet)
- **ConvNeXt-Tiny** — timm (`convnext_tiny.fb_in22k_ft_in1k`)
- **SwinV2-Tiny** — timm (`swinv2_cr_tiny_ns_224`)
- **ViT-L/16** — timm (`vit_large_patch16_224`)
- **Phikon-V2** — HuggingFace (`owkin/phikon-v2`), DINOv2 backbone pretrained on histopathology

## Handling Class Imbalance

Four optimization strategies are available:

| Strategy | Flag | Method |
|----------|------|--------|
| Baseline | `--optimization baseline` | Standard CrossEntropyLoss |
| Weighted Sampler | `--optimization weighted_sampler` | Oversample minority classes |
| Weighted CE | `--optimization weighted_ce` | Inverse-frequency class weights |
| LDAM | `--optimization ldam` | Label-Distribution-Aware Margin Loss (Cao et al. 2019) with Deferred Re-Weighting |

## Data Augmentation

| Strategy | Flag | Description |
|----------|------|-------------|
| Baseline | `--augmentation baseline` | Random flip + rotation + color jitter |
| HistoRandAugment | `--augmentation randaugment` | 16 histopathology-specific operations with PAS stain decomposition (Faryna et al. 2021/2024) |
| Manual | `--augmentation manual` | Morphological augmentations following Tellez et al. 2019 |

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.10+, CUDA GPU, and dependencies listed in `requirements.txt`.

## Usage

### Training

```bash
# baseline ResNet-50
glom-train --model resnet50 --optimization baseline --augmentation baseline

# Phikon-V2 with LDAM + HistoRandAugment
glom-train --model phikon_v2 --optimization ldam --augmentation randaugment \
    --lr 1e-4 --epochs 50 --batch-size 32

# with backbone freezing (two-phase training)
glom-train --model swinv2 --freeze-backbone-epochs 5 --discriminative-lr

# from YAML config
glom-train --config configs/base.yaml --model convnext
```

Key training flags:

```
--epochs N              Number of training epochs (default: 50)
--lr FLOAT              Learning rate (default: 1e-4)
--batch-size N          Batch size (default: 32)
--early-stopping N      Patience (default: 15)
--accumulation-steps N  Gradient accumulation (default: 1)
--freeze-backbone-epochs N  Phase-1 frozen backbone epochs
--discriminative-lr     Lower LR for backbone layers
--ldam-drw-start FLOAT  DRW activation point as epoch fraction (default: 0.5)
```

### Evaluation

```bash
glom-evaluate --checkpoint outputs/checkpoints/best.pt \
    --test-dir data_split_patient/test \
    --model resnet50 \
    --output-dir outputs/results
```

Produces `test_metrics.json`, `confusion_matrix.png`, `classification_report.txt`, `predictions.json`, and `misclassified.json`.

### CAM Visualization

```bash
glom-cam --checkpoint outputs/checkpoints/best.pt \
    --data-dir data_split_patient/test \
    --model resnet50 \
    --methods gradcam gradcam++ eigencam \
    --max-images 50

# attention rollout for ViTs
glom-cam --model vit_large --methods attention_rollout \
    --checkpoint best.pt --data-dir data_split_patient/test
```

Supported methods: `gradcam`, `gradcam++`, `eigencam`, `layercam`, `scorecam`, `hirescam`, `attention_rollout`.

## Project Structure

```
src/
├── cli/                # Command-line entry points
│   ├── train.py        # Training
│   ├── evaluate.py     # Test evaluation
│   └── generate_cam.py # CAM visualization
├── data/
│   ├── dataset.py      # GlomerularDataset (folder-based)
│   └── transforms.py   # Augmentation pipelines + HistoRandAugment
├── models/
│   ├── factory.py      # Model creation for all 5 architectures
│   └── phikon.py       # Phikon-V2 wrapper
├── training/
│   ├── trainer.py      # Training loop (AMP, early stopping, DRW)
│   ├── losses.py       # LDAMLoss + loss factory
│   ├── samplers.py     # WeightedRandomSampler
│   └── config.py       # TrainingConfig dataclass
├── evaluation/
│   ├── metrics.py      # MetricsSet + compute_all_metrics
│   ├── cam.py          # GradCAM + AttentionRollout
│   └── test_evaluation.py  # TestEvaluator
└── utils/
    └── logging.py      # Logging setup
configs/
    base.yaml           # Default hyperparameters
```

## Data Format

Images should be organized in class folders:

```
data_split_patient/
├── train/
│   ├── Pattern1_Unauffälliger Glomerulus/
│   │   ├── image001.png
│   │   └── ...
│   ├── Pattern2_Amyloidose/
│   └── ...
├── val/
└── test/
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.pt` (pre-processed tensors).

## References

- Cao, K., Wei, C., Gargade, A., Yuille, A., & Fidler, S. (2019). Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss. *NeurIPS*.
- Faryna, K., van der Laak, J., & Litjens, G. (2021/2024). Tailoring automated data augmentation to H&E-stained histopathology. *MIDL*.
- Tellez, D., Litjens, G., Bándi, P., et al. (2019). Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. *Medical Image Analysis*.
- Filiot, A., Gherber, H., Olivier, A., et al. (2023). Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling. *medRxiv* (Phikon-V2).

## License

MIT

