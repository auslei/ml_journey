"""Pretrained torchvision models, cached outside git, with a clean fallback."""

import logging
import os

import torch
from torchvision import models

from common.datasets import DATA_DIR

logger = logging.getLogger(__name__)


def load_resnet18(
    num_classes: int, pretrained: bool = True, seed: int = 42
) -> tuple[torch.nn.Module, bool]:
    """Return ``(model, used_pretrained)``: ResNet18 with a fresh ``num_classes`` head.

    Weights download once into ``data/torch`` (gitignored). If ``pretrained`` is True and the
    download fails, the model falls back to random initialisation and ``used_pretrained`` is
    False, so the caller can say so. The new head is seeded.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be at least 1, got {num_classes}")
    torch.hub.set_dir(str(DATA_DIR / "torch"))
    os.makedirs(DATA_DIR / "torch", exist_ok=True)
    used = False
    model = None
    if pretrained:
        logger.info("Loading pretrained ResNet18 weights")
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            used = True
        except Exception as err:  # noqa: BLE001  (any download or cache failure falls back)
            logger.warning("Pretrained weights unavailable (%s). Using random init.", err)
    if model is None:
        torch.manual_seed(seed)
        model = models.resnet18(weights=None)
    torch.manual_seed(seed)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model, used
