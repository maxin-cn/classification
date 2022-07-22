# Copyright (c) Meituan.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.quantization.default_mappings import DEFAULT_QAT_MODULE_MAPPING
from torch.quantization import *
from collections import OrderedDict

def qat_prepare(net):
    _quant_min_w = -127
    _quant_max_w = 127
    _qtype_w = torch.qint8
    _qscheme_w = torch.per_channel_symmetric
    _reduce_range_w = False

    _quant_min_a = -127
    _quant_max_a = 127
    _qtype_a = torch.qint8
    _qscheme_a = torch.per_tensor_symmetric
    _reduce_range_a = False

    weight_fake_quant_trt = FakeQuantize.with_args(observer=PerChannelMinMaxObserver,
                                                    quant_min = _quant_min_w,
                                                    quant_max = _quant_max_w,
                                                    dtype = _qtype_w,
                                                    qscheme = _qscheme_w,
                                                    reduce_range = _reduce_range_w,
                                                    ch_axis = 0)

    act_fake_quant_trt    = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                    quant_min = _quant_min_a,
                                                    quant_max = _quant_max_a,
                                                    dtype = _qtype_a,
                                                    qscheme = _qscheme_a,
                                                    reduce_range = _reduce_range_a)

    net.qconfig = QConfig(activation = act_fake_quant_trt, weight = weight_fake_quant_trt) 
    torch.quantization.prepare_qat(net, inplace=True)
    net.eval()
    
    print('QAT config: weights: _min: %d | _max: %d | dtype: %s | qscheme: %s | if_reduce: %s' %(_quant_min_w, _quant_max_w, _qtype_w, _qscheme_w, _reduce_range_w))
    print('QAT config: activations: _min: %d | _max: %d | dtype: %s | qscheme: %s | if_reduce: %s' %(_quant_min_a, _quant_max_a, _qtype_a, _qscheme_a, _reduce_range_a))

    return net

def freeze_quantizer(net):
    ## Freeze quantizer parameters
    net.apply(torch.quantization.disable_observer)

def freeze_bn(net):   
    ## Freeze batch norm mean and variance estimates
    net.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

def fuse_net(net):
    for m in net.modules():
        if type(m) == torch.nn.modules.container.Sequential:
            for idx in range(len(m)):
                if type(m[idx]) == nn.Conv2d:
                    torch.quantization.fuse_modules(m, [str(idx), str(idx + 1)], inplace=True)
        """
        if type(m) == InvertedResidual:
            for idx in range(len(m.conv)):
                if type(m.conv[idx]) == nn.Conv2d:
                    torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)
        """
    print("fuse finish")