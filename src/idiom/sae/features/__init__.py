"""The per-residue feature-activation dataset: building it, reading it, and enrichment analysis."""

from idiom.sae.features.build_feature_dataset import build_feature_dataset
from idiom.sae.features.feature_activations import FeatureDataset

__all__ = ["FeatureDataset", "build_feature_dataset"]
