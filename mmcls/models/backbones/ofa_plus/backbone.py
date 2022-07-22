# Copyright (c) Meituan. All rights reserved.
import torch.nn as nn

from ...builder import BACKBONES
from .model_zoo import ResNetSS
from ..base_backbone import BaseBackbone
import json
import os
import sys


@BACKBONES.register_module()
class OFAPlus(BaseBackbone):
    """ OFAPlus backbone.

    A set of ResNet Backbones in the ResNet50D's search space.

    Args:
        config_name(key): The specialized OFAPlus net based on net_config.
        ...
    """

    def __init__(self, config_name, **kwargs):
        super(OFAPlus, self).__init__(**kwargs)
        cur_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(cur_path, "net_config")
        self.config_name = config_name
        self.net_config = json.load(open(os.path.join(config_path, self.config_name+".config"), 'r'))
        self.net = ResNetSS.build_from_config(self.net_config)

    def forward(self, x):
        return self.net(x)

    def init_weights(self, pretrained=False):
        pass

    def train(self, mode=True):
        super(OFAPlus, self).train(mode)
