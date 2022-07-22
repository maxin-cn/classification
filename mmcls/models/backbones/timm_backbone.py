# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import torch.nn as nn
import logging

try:
    import timm
except ImportError:
    timm = None

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .timm_custom import *

@BACKBONES.register_module()
class TIMMBackbone(BaseBackbone):
    """Wrapper to use backbones from timm library. More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_ .

    Args:
        model_name (str): Name of timm model to instantiate.
        pretrained (bool): Load pretrained weights if True.
        checkpoint_path (str): Path of checkpoint to load after
            model is initialized.
        in_channels (int): Number of input image channels. Default: 3.
        init_cfg (dict, optional): Initialization config dict
        **kwargs: Other timm & model specific arguments.
    """

    def __init__(
        self,
        model_name,
        pretrained=False,
        checkpoint_path='',
        in_channels=3,
        init_cfg=None,
        **kwargs,
    ):
        if timm is None:
            raise RuntimeError('timm is not installed')
        super(TIMMBackbone, self).__init__(init_cfg)
        self.timm_model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        self.num_features = self.timm_model.num_features

        # Make unused parameters None
        self.timm_model.global_pool = None
        self.timm_model.fc = None
        self.timm_model.classifier = None

    def forward(self, x):
        features = self.timm_model.forward_features(x)
        return (features, )

    def init_weights(self):
        if not hasattr(self, 'init_cfg'):
            return
        if self.init_cfg is None:
            return
        if 'type' not in self.init_cfg:
            return
        if self.init_cfg['type'] == 'Pretrained':
            logger = logging.getLogger()
            if 'checkpoint' in self.init_cfg:
                model_url  =  self.init_cfg['checkpoint']
            else:
                model_url = self.timm_model.default_cfg['url']
            _my_load_checkpoint(self, model_url, strict=False, logger=logger)
        elif self.init_cfg['type'] is None:
            from mmcv.cnn import constant_init, kaiming_init
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


def norm_weight(m):
    " Normalize StdConv2d weight on loading checkpoint"
    classname = m.__class__.__name__
    if classname.find('StdConv2d') != -1:
        std = torch.std(m.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        mean = torch.mean(m.weight, dim=[1, 2, 3], keepdim=True)
        m.weight = nn.Parameter((m.weight - mean) / (std + m.eps))


def _my_load_checkpoint(model,
                        filename,
                        map_location=None,
                        strict=False,
                        logger=None,
                        prefix='timm_model.',
                        load_head=False):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    backbone = model.backbone if hasattr(model, 'backbone') else model

    # use custom load_pretrained if in npy or npz
    if os.path.splitext(filename)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(backbone.timm_model, 'load_pretrained'):
            backbone.timm_model.load_pretrained(filename)
            backbone.timm_model.apply(norm_weight)
            return None

    from mmcv.runner import _load_checkpoint, load_state_dict
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    state_dict = {prefix + k: v for k, v in state_dict.items()}

    # load state_dict
    load_state_dict(backbone, state_dict, True, logger)

    if load_head and hasattr(model, 'head'):
        if prefix+'fc.weight' in state_dict:
            model.head.fc.weight.data = state_dict[prefix+'fc.weight']
            model.head.fc.bias.data = state_dict[prefix+'fc.bias']
        elif prefix+'head.fc.weight' in state_dict:
            model.head.fc.weight.data = state_dict[prefix+'head.fc.weight'].flatten(1)
            model.head.fc.bias.data = state_dict[prefix+'head.fc.bias']
        elif prefix+'head.weight' in state_dict:
            model.head.fc.weight.data = state_dict[prefix+'head.weight']
            model.head.fc.bias.data = state_dict[prefix+'head.bias']
        else:
            assert False, "cannot load head weights"

    return checkpoint
