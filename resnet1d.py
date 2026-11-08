"""Small 1D ResNet for PTB-XL superclass classification (multi-label, 5 classes).

Owner: Machine learning. Train on clean data first (experiment 1), then test
on corrupted data (2), then retrain per AFE config (3).
"""

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def build_resnet1d(in_leads: int = 1, n_classes: int = len(CLASSES)):
    raise NotImplementedError("TODO (ML): implement with torch.nn; keep it small.")
