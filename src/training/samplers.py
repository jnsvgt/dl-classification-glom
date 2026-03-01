"""Weighted random sampler for class-imbalanced datasets."""

from torch.utils.data import WeightedRandomSampler


def create_weighted_sampler(dataset):
    """Build a sampler that over-samples minority classes."""
    class_weights = dataset.get_class_weights()
    sample_weights = [class_weights[label].item() for _, label in dataset.samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
