# Copyright (c) Meituan. All rights reserved.
import os
import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

try:
    import timm
except ImportError:
    timm = None

from fvcore.nn import flop_count, parameter_count
from mmcls.models.backbones import TIMMBackbone


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_timm_all_backbones(capsys):
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = TIMMBackbone()
        model.init_weights(pretrained=0)

    if timm is not None:
        for model_name in timm.list_models("*twins*"):
            model = TIMMBackbone(model_name)
            model.init_weights()
            model.train()
            assert check_norm_state(model.modules(), True)

            imgs = torch.randn(1, 3, 224, 224)
            feat = model(imgs)
            assert len(feat) == 1
            with capsys.disabled():
                # print(model_name, feat[0].shape)
                gflops, unsupported = flop_count(model, inputs=(imgs,))
                params = parameter_count(model)
                # print("GFLOPS:", sum(gflops.values()), "MParams:", params[""]/1e6)
                print(model_name, params[""]/1e6, sum(gflops.values()))

        for model_name in timm.list_models("*efficientnet*"):
            model = TIMMBackbone(model_name)
            model.init_weights()
            model.train()
            assert check_norm_state(model.modules(), True)

            imgs = torch.randn(1, 3, 224, 224)
            feat = model(imgs)
            assert len(feat) == 1
            with capsys.disabled():
                # print(model_name, feat[0].shape)
                gflops, unsupported = flop_count(model, inputs=(imgs,))
                params = parameter_count(model)
                # print("GFLOPS:", sum(gflops.values()), "MParams:", params[""] / 1e6)
                print(model_name, params[""] / 1e6, sum(gflops.values()))


@pytest.mark.skip(reason="temporarily skip testing this (some tf_efficientnet models requires dynamic F.pad)")
@pytest.mark.xfail(raises=Exception)
def test_timm_onnx_export(capsys):
    if timm:
        tmp_dir = '.pytest_cache/onnx'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        models_to_test = timm.list_models("*twins*") + timm.list_models("*efficientnet*")
        for model_name in models_to_test:
            model_path = os.path.join(tmp_dir, model_name + ".onnx")
            if os.path.exists(model_path):
                continue
            with capsys.disabled():
                print("exporting", model_name, "to", model_path)

            model = TIMMBackbone(model_name)
            model.init_weights()
            model.train()
            # assert check_norm_state(model.modules(), True)
            imgs = torch.randn(1, 3, 224, 224)
            with pytest.raises(Exception):
                torch.onnx.export(model, imgs, model_path,
                                  input_names=['input'],
                                  output_names=['output'],
                                  verbose=False,
                                  opset_version=11,
                                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

                assert(os.path.exists(model_path))
                with capsys.disabled():
                    print("exported:", model_name, "to", model_path)

                # os.remove(model_path)
