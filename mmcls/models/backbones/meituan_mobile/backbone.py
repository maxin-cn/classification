# Copyright (c) Meituan. All rights reserved.
import torch.nn as nn

from ...builder import BACKBONES
from .model_zoo import MobileNetV3SS
from ..base_backbone import BaseBackbone
import json
import os
import sys
import logging


@BACKBONES.register_module()
class MeituanMobile(BaseBackbone):
    """ MeituanMobile backbone.

    A set of Mobile Backbones in the MobileNetV3's search space.

    Args:
        config_name(key): The specialized MeituanMobile net based on net_config.
        ...
    """

    def __init__(self, config_name, **kwargs):
        super(MeituanMobile, self).__init__(**kwargs)
        cur_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(cur_path, "net_config")
        self.config_name = config_name
        self.net_config = json.load(open(os.path.join(config_path, self.config_name+".config"), 'r'))
        self.net = MobileNetV3SS.build_from_config(self.net_config)
        self.num_features = self.net.feature_mix_layer.out_channels

    def forward(self, x):
        return self.net(x)

    def init_weights(self):
        if not hasattr(self, 'init_cfg'):
            return
        if self.init_cfg is None:
            return
        if 'type' not in self.init_cfg:
            return
        if self.init_cfg['type'] == 'Pretrained':
            if 'checkpoint' in self.init_cfg:
                model_url  =  self.init_cfg['checkpoint']
            else:
                return
        from mmcv.runner import _load_checkpoint, load_state_dict
        checkpoint = _load_checkpoint(model_url, map_location=None)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        logger = logging.getLogger()
        load_state_dict(self.net, state_dict, False, logger)

    def train(self, mode=True):
        super(MeituanMobile, self).train(mode)
