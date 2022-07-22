import torch
import torch.nn as nn
from functools import partial

from timm.models.registry import register_model
from timm.models.resnetv2 import _create_resnetv2
from timm.models.twins import _create_twins

from .layers import StdConv2d


def _create_resnetv2_bit(variant, pretrained=False, **kwargs):
    return _create_resnetv2(
        variant, pretrained=pretrained, stem_type='', conv_layer=partial(StdConv2d, eps=1e-8), **kwargs)


@register_model
def resnetv2_50x1_bitm_fix(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_50x1_bitm', pretrained=pretrained, layers=[3, 4, 6, 3], width_factor=1, **kwargs)


@register_model
def resnetv2_101x1_bitm_fix(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_101x1_bitm', pretrained=pretrained, layers=[3, 4, 23, 3], width_factor=1, **kwargs)

@register_model
def resnetv2_152x2_bitm_fix(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_152x2_bitm', pretrained=pretrained, layers=[3, 8, 36, 3], width_factor=2, **kwargs)


@register_model
def resnetv2_152x2_bitm_in21k_fix(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_152x2_bitm_in21k', pretrained=pretrained, num_classes=kwargs.pop('num_classes', 21843),
        layers=[3, 8, 36, 3], width_factor=2, **kwargs)


@register_model
def resnetv2_50x1_bit_distilled_fix(pretrained=False, **kwargs):
    """ ResNetV2-50x1-BiT Distilled
    Paper: Knowledge distillation: A good teacher is patient and consistent - https://arxiv.org/abs/2106.05237
    """
    return _create_resnetv2_bit(
        'resnetv2_50x1_bit_distilled', pretrained=pretrained, layers=[3, 4, 6, 3], width_factor=1, **kwargs)


@register_model
def resnetv2_152x2_bit_teacher_fix(pretrained=False, **kwargs):
    """ ResNetV2-152x2-BiT Teacher
    Paper: Knowledge distillation: A good teacher is patient and consistent - https://arxiv.org/abs/2106.05237
    """
    return _create_resnetv2_bit(
        'resnetv2_152x2_bit_teacher', pretrained=pretrained, layers=[3, 8, 36, 3], width_factor=2, **kwargs)


@register_model
def resnetv2_152x2_bit_teacher_384_fix(pretrained=False, **kwargs):
    """ ResNetV2-152xx-BiT Teacher @ 384x384
    Paper: Knowledge distillation: A good teacher is patient and consistent - https://arxiv.org/abs/2106.05237
    """
    return _create_resnetv2_bit(
        'resnetv2_152x2_bit_teacher_384', pretrained=pretrained, layers=[3, 8, 36, 3], width_factor=2, **kwargs)


@register_model
def twins_svt_exlarge(pretrained=False, **kwargs):
    """"
    To Match Swin-Large @ 224x224
    """
    model_kwargs = dict(
        patch_size=4, embed_dims=[192, 384, 768, 1536], num_heads=[6, 12, 24, 48], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
    return _create_twins('twins_svt_large', pretrained=pretrained, **model_kwargs)
