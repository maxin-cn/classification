# -*- coding: UTF-8 -*-
import argparse
from tools import *

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMCls to ONNX')
    parser.add_argument('--preprocess-weight', help='preprocess weight', default="https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/scripts/preprocess_mmclassification.py")
    parser.add_argument('--model-weight', help='model weight', default="tmp.onnx")
    parser.add_argument('--fp16', action='store_true', help='Enable fp16 mode')
    parser.add_argument('--int8', action='store_true', help='Enable int8 mode')
    parser.add_argument('--int8-mode', help='int8 mode', default="Simple")
    parser.add_argument('--int8-calibration', help='int8 calibration table', default="")
    parser.add_argument('--appkey', help='serving appkey', default="com.sankuai.basecv.serving.autodeploy")
    parser.add_argument('--misid', help='misid', default="lishengxi")
    parser.add_argument('--project', help='project', default="platcv")
    parser.add_argument('--cluster', help='serving cluster', default="GH")
    parser.add_argument('--queue', help='serving queue', default="root.gh_serving.hadoop-vision.serving")
    parser.add_argument('--custom-input', action='store_true', help='Enable custom input')
    parser.add_argument('--input-shape', nargs='+', default=[4, 3, 224, 224], type=int, help='Specify input')
    parser.add_argument('--output-shape', nargs='+', default=0, type=int, help='specify output')
    parser.add_argument('--input-shape-type', default='NCHW', help='Specify input type: NCHW or NHWC')
    parser.add_argument('--device-type', default='gpu', help='Specify device type: cpu or gpu')
    parser.add_argument('--network', default='resnet50', help='Specify network name')
    parser.add_argument('--hadoop-name', default='hadoop-platcv', help='Specify hadoop name')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    params = dict(
        preprocess_weight=args.preprocess_weight,
        weight=args.model_weight,
        setFp16=args.fp16,
        setInt8=args.int8,
        int8Mode=args.int8_mode,
        dynamicRangeFileName=args.int8_calibration,
        appkey=args.appkey,
        misid=args.misid,
        project=args.project,
        cluster=args.cluster,
        queue=args.queue,
        custom_input=args.custom_input,
        input_shape=args.input_shape,
        input_shape_type=args.input_shape_type,
        output_shape=args.output_shape,
        device_type=args.device_type,
        network=args.network,
        hadoop_name=args.hadoop_name)

    try:
        print("模型部署参数:{}".format(params))
        print("开始使用efficient serving部署模型")
        response = submit_efficient_serving(params, params)
        print("部署结果:{}".format(response))
    except NameError:
        print("模型部署失败")






