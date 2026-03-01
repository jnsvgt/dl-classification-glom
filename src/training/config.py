"""Training configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    # model
    model_name: str = "resnet50"
    num_classes: int = 9

    # optimization
    optimization: str = "baseline"  # baseline, weighted_sampler, weighted_ce, ldam
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    batch_size: int = 32
    accumulation_steps: int = 1
    warmup_epochs: int = 5
    early_stopping_patience: int = 15
    label_smoothing: float = 0.0
    gradient_clip_norm: Optional[float] = None

    # augmentation
    augmentation: str = "baseline"

    # backbone freeze (two-phase)
    freeze_backbone_epochs: int = 0
    phase2_warmup_epochs: int = 2
    discriminative_lr_enabled: bool = False
    backbone_lr_factor: float = 0.1

    # LDAM
    ldam_drw_start: float = 0.5  # fraction of total epochs
    ldam_max_margin: float = 0.5

    # data
    data_dir: str = "data_split_patient"
    num_workers: int = 4
    seed: int = 42

    # output
    checkpoint_dir: str = "outputs/checkpoints"
    results_dir: str = "outputs/results"

    # internal (set at runtime, not by user)
    _class_counts: Optional[list] = field(default=None, repr=False)
    _class_names: Optional[list] = field(default=None, repr=False)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d):
        import dataclasses
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})
