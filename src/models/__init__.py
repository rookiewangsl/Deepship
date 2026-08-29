"""Model definitions for ship-noise classification."""

from .ma_cnn_a import (
    MACNNAClassifier,
    MACNNATemporalClassifier,
    build_macnna_model,
)
from .wav2vec2_conformer import Wav2Vec2ConformerClassifier

__all__ = [
    "MACNNAClassifier",
    "MACNNATemporalClassifier",
    "Wav2Vec2ConformerClassifier",
    "build_macnna_model",
]
