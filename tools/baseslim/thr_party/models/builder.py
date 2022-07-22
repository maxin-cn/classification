# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

ALGORITHMS = MODELS
DISTILLERS = MODELS
LOSSES = MODELS
ARCHITECTURES = MODELS

def build_algorithm(cfg):
    """Build compressor."""
    return ALGORITHMS.build(cfg)


def build_architecture(cfg):
    """Build architecture."""
    return ARCHITECTURES.build(cfg)


def build_distiller(cfg):
    """Build distiller."""
    return DISTILLERS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)
