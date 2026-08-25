"""Model definitions for ship-noise classification."""

from .ma_cnn_a import MACNNAClassifier
from .wav2vec2_conformer import Wav2Vec2ConformerClassifier

__all__ = ["MACNNAClassifier", "Wav2Vec2ConformerClassifier"]
