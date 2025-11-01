"""Simple late fusion used only to mimic the behaviour from the notebook.

This is *not* a full CNN: we assume that imaging features are already extracted
(e.g. from a pretrained ResNet-18) and we combine them with tabular features.
"""

import numpy as np


class LateFusionModel:
    def __init__(self, imaging_weight: float = 0.25):
        self.imaging_weight = imaging_weight

    def predict_proba(self, tabular_probs: np.ndarray, imaging_probs: np.ndarray) -> np.ndarray:
        """Fuse two probability vectors (N, 2) coming from tabular and imaging models."""
        if tabular_probs.shape != imaging_probs.shape:
            raise ValueError("Tabular and imaging probabilities must have the same shape.")

        fused = (1.0 - self.imaging_weight) * tabular_probs + self.imaging_weight * imaging_probs
        return fused
