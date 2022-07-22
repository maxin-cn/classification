# -*- coding: UTF-8 -*-
import argparse
from functools import partial

import mmcv
import numpy as np
import onnxruntime as rt
import torch
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint

from mmcls.models import build_classifier

from pytorch2onnx import *
from tools import *


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMCls to ONNX')
    parser.add_argument('--model-infos', help='test config and checkpoint file path', default=None)
    parser.add_argument('--config', help='test config file path', default=None)
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export ONNX with dynamic input shape. \
            Defaults to False.')
    args = parser.parse_args()
    return args

def convert(args):
    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    classifier = build_classifier(cfg.model)

    if args.checkpoint:
        load_checkpoint(classifier, args.checkpoint, map_location='cpu')

    # convert model to onnx file
    pytorch2onnx(
        classifier,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        dynamic_export=args.dynamic_export,
        output_file=args.output_file,
        do_simplify=args.simplify,
        verify=args.verify)


if __name__ == '__main__':
    args = parse_args()
    model_infos, numbers = read_text(args.model_infos)
    convert_infos = list()
    t1 = time.time()
    for i in range(numbers):
        model_info = model_infos[i].split("\t")
        print("正在处理第{}个模型，模型结构为{}".format(i+1, model_info[0]))
        args.config = model_info[1]
        args.checkpoint = model_info[2]
        args.output_file = args.checkpoint.replace(".pth", ".onnx")
        args.shape = [int(model_info[3]), int(model_info[3])]
        convert_status = dict(modelname=model_info[0], pytorch2onnx="ok", consistency=True)
        try:
            convert(args)
        except AttributeError:
            convert_status["pytorch2onnx"] = "fail"
        except ValueError:
            convert_status["pytorch2onnx"] = "fail"
            convert_status["consistency"] = False
        convert_infos.append(convert_status)
    t2 = time.time()
    print("全部处理完成，共耗时{}s".format(int(t2-t1)))
    print(convert_infos)


