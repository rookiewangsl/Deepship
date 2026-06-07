"""Model definitions for ship-noise classification."""

from .ma_cnn_a import MACNNAClassifier
from .mel_cnn import MelCNNClassifier
from .waveform_transformer import STFTMAEPretrainer, STFTTransformerClassifier

__all__ = [
    "MACNNAClassifier",
    "MelCNNClassifier",
    "STFTMAEPretrainer",
    "STFTTransformerClassifier",
]
