# Copyright (c) Meituan. All rights reserved.
import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from fvcore.nn import flop_count, parameter_count
from mmcls.models.backbones import MeituanMobile


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_meituan_mobile_backbone(capsys):
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = MeituanMobile()
        model.init_weights(pretrained=True)

    # Test a model
    _configs = ['meituan_cv_acc70.4_flops71M',
                'meituan_cv_acc70.5_flops71M',
                'meituan_cv_acc71.1_flops78M',
                'meituan_cv_acc71.4_flops81M',
                'meituan_cv_acc71.4_flops81M_v2',
                'meituan_cv_acc71.5_flops84M',
                'meituan_cv_acc72.8_flops106M',
                'meituan_cv_acc73.0_flops109M',
                'meituan_cv_acc73.1_flops102M',
                'meituan_cv_acc73.3_flops115M',
                'meituan_cv_acc73.4_flops119M',
                'meituan_cv_acc73.6_flops116M',
                'meituan_cv_acc74.7_flops154M',
                'meituan_cv_acc74.7_flops160M',
                'meituan_cv_acc74.7_flops176M',
                'meituan_cv_acc74.9_flops171M',
                'meituan_cv_acc74.9_flops173M',
                'meituan_cv_acc75.5_flops172M',
                'meituan_cv_acc75.8_flops220M',
                'meituan_cv_acc76.1_flops232M',
                'meituan_cv_acc76.3_flops231M',
                'meituan_cv_acc76.4_flops243M',
                'meituan_cv_acc76.6_flops251M',
                'meituan_cv_acc76.9_flops243M',
                'meituan_cv_acc78.4_flops357M',
                'meituan_cv_acc78.7_flops373M',
                'meituan_cv_acc79.3_flops479M',
                'meituan_cv_acc79.7_flops581M',
                'meituan_cv_acc79.8_flops620M',
                'meituan_cv_acc80.1_flops672M',
                'meituan_cv_acc80.2_flops779M']

    for config in _configs:
        model = MeituanMobile(config_name=config)
        model.train()
        assert check_norm_state(model.modules(), True)

        imgs = torch.randn(1, 3, 224, 224)

        with capsys.disabled():
            gflops, unsupported = flop_count(model, inputs=(imgs,))
            params = parameter_count(model)
            print(config, params[""] / 1e6, sum(gflops.values()))
