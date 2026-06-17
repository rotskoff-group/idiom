"""Feature interpretation: build the per-residue feature-activation dataset and read/reduce it."""

from idiom.sae.features.build_feature_dataset import build_feature_dataset
from idiom.sae.features.feature_activations import FeatureDataset

__all__ = ["FeatureDataset", "build_feature_dataset"]
