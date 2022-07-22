# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import numpy as np

from onnx2tensorrt import *
from tools import *


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMClassification models from ONNX to TensorRT')
    parser.add_argument('--model-infos', help='test config and checkpoint file path', default=None)
    parser.add_argument(
        '--onnx-file',
        type=str,
        default='tmp.onnx',
        help='Filename of the input ONNX model')
    parser.add_argument(
        '--trt-file',
        type=str,
        default='tmp.trt',
        help='Filename of the output TensorRT engine')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the outputs of ONNXRuntime and TensorRT')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='Input size of the model')
    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=1,
        help='Maximum batch size of TensorRT model.')
    parser.add_argument('--fp16', action='store_true', help='Enable fp16 mode')
    parser.add_argument(
        '--workspace-size',
        type=int,
        default=1,
        help='Max workspace size of GPU in GiB')
    args = parser.parse_args()
    return args

def convert(args):
    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    # Create TensorRT engine
    onnx2tensorrt(
        args.onnx_file,
        args.trt_file,
        input_shape,
        args.max_batch_size,
        fp16_mode=args.fp16,
        verify=args.verify,
        workspace_size=args.workspace_size)

if __name__ == '__main__':
    args = parse_args()
    model_infos, numbers = read_text(args.model_infos)
    convert_infos = list()
    t1 = time.time()
    for i in range(numbers):
        model_info = model_infos[i].split("\t")
        print("正在处理第{}个模型，模型结构为{}".format(i+1, model_info[0]))
        args.onnx_file = model_info[2].replace(".pth", ".onnx")
        args.trt_file = model_info[2].replace(".pth", ".trt")
        args.shape = [int(model_info[3]), int(model_info[3])]
        convert_status = dict(modelname=model_info[0], onnx2tensorrt="ok", consistency=True)
        try:
            convert(args)
        except ValueError:
            convert_status["onnx2tensorrt"] = "fail"
        convert_infos.append(convert_status)
    t2 = time.time()
    print("全部处理完成，共耗时{}s".format(int(t2-t1)))
    print(convert_infos)
