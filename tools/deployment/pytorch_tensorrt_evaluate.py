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
from mmcv.tensorrt import TRTWraper
import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMCls to ONNX')
    parser.add_argument('--model-infos', help='test config and checkpoint file path', default=None)
    parser.add_argument('--config', help='test config file path', default=None)
    parser.add_argument('--pretrained', help='pretrained', action='store_true')
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

def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_labels = rng.randint(
        low=0, high=num_classes, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(False),
        'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs

def evaluate(args, evaluate_status):
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
    pytorch_model = build_classifier(cfg.model)
    if args.checkpoint:
        from mmcls.models.backbones.timm_backbone import _my_load_checkpoint, TIMMBackbone
        if type(pytorch_model.backbone) is TIMMBackbone and args.pretrained:
            checkpoint = _my_load_checkpoint(pytorch_model, args.checkpoint, map_location='cpu', prefix='timm_model.', load_head=True)
        else:
            checkpoint = load_checkpoint(pytorch_model, args.checkpoint, map_location='cuda:0')
    pytorch_model.cuda().eval()
    if hasattr(pytorch_model.head, 'num_classes'):
        num_classes = pytorch_model.head.num_classes
    # Some backbones use `num_classes=-1` to disable top classifier.
    elif getattr(pytorch_model.backbone, 'num_classes', -1) > 0:
        num_classes = pytorch_model.backbone.num_classes
    else:
        raise AttributeError('Cannot find "num_classes" in both head and '
                             'backbone, please check the config file.')

    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_list = [img[None, :].cuda() for img in imgs]
    
    loop_times = 1000
    # warmup
    for i in range(10):
        with torch.no_grad():pytorch_result = pytorch_model(img_list, img_metas={}, return_loss=False)[0]

    t1 = time.time()
    torch.cuda.current_stream().synchronize()
    for i in range(loop_times):
        with torch.no_grad():pytorch_result = pytorch_model(img_list, img_metas={}, return_loss=False)[0]
    torch.cuda.current_stream().synchronize()
    t2 = time.time()
    pytorch_time_used = (t2-t1)/loop_times*1000
    evaluate_status["pytorch_speed"] = pytorch_time_used
    print("pytorch_model predict time:", pytorch_time_used)  

    # Get results from TensorRT
    input_names = ['input']
    output_names = ['probs']
    trt_model = TRTWraper(args.trt_file, input_names, output_names)
    for i in range(10):
        with torch.no_grad():trt_outputs = trt_model({input_names[0]: img_list[0]})

    t1 = time.time()
    torch.cuda.current_stream().synchronize()
    for i in range(loop_times):
        with torch.no_grad():trt_outputs = trt_model({input_names[0]: img_list[0]})
        trt_outputs = [
            trt_outputs[_].detach().cpu().numpy() for _ in output_names
        ]
    torch.cuda.current_stream().synchronize()
    t2 = time.time()
    tensorrt_time_used = (t2-t1)/loop_times*1000
    evaluate_status["tensorrt_speed"] = tensorrt_time_used
    print("tensorrt_model predict time:", tensorrt_time_used) 

    evaluate_status["speedup_ratio"] = pytorch_time_used / tensorrt_time_used - 1

    diff = np.mean(np.abs(pytorch_result-trt_outputs[0][0]))
    consistency = np.allclose(pytorch_result, trt_outputs[0][0], rtol=1e-03, atol=5e-03)
    evaluate_status["consistency"] = consistency 
    evaluate_status["diff"] = diff.astype(float)
    print("consistency:", consistency)
    print("diff:", diff) 

    return evaluate_status


    

if __name__ == '__main__':
    args = parse_args()
    model_infos, numbers = read_text(args.model_infos)
    evaluate_infos = list()
    t1 = time.time()
    for i in range(numbers):
        model_info = model_infos[i].split("\t")
        print("正在处理第{}个模型，模型结构为{}".format(i+1, model_info[0]))
        args.config = model_info[1]
        args.checkpoint = model_info[2]
        if model_info[2].endswith(".pth"):
            args.trt_file = model_info[2].replace(".pth", ".trt")
        else:
            trt.init_libnvinfer_plugins(None, "")
            args.trt_file = model_info[2].replace(".npz", ".trt")
        args.shape = [int(model_info[3]), int(model_info[3])]
        evaluate_status = dict(modelname=model_info[0], status=True, consistency=True, 
                    pytorch_speed=0, tensorrt_speed=0, speedup_ratio=0)
        try:
            evaluate_status = evaluate(args, evaluate_status)
        except:
            evaluate_status["status"] = False
        print("evaluate_status:", evaluate_status)
        evaluate_infos.append(evaluate_status)
    t2 = time.time()
    print("全部处理完成，共耗时{}s".format(int(t2-t1)))
    print(evaluate_infos)
    
    with open("model_speed.json", "w") as f:
        json.dump(evaluate_infos, f)



