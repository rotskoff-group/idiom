"""Sparse feature datasets, peak windows, enrichment, and reusable design signatures."""

from idiom.sae.features.build_feature_dataset import build_feature_dataset
from idiom.sae.features.enrichment import (
    enrich,
    feature_counts,
    load_enrichment,
    save_enrichment,
    select_features,
)
from idiom.sae.features.feature_dataset import FeatureDataset
from idiom.sae.features.signatures import combine_signatures, load_signatures, write_signature
from idiom.sae.features.windows import AMINO_ACIDS, feature_windows, logo_data

__all__ = [
    "AMINO_ACIDS",
    "FeatureDataset",
    "build_feature_dataset",
    "combine_signatures",
    "enrich",
    "feature_counts",
    "feature_windows",
    "load_enrichment",
    "load_signatures",
    "logo_data",
    "save_enrichment",
    "select_features",
    "write_signature",
]
