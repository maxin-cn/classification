# -*- coding: UTF-8 -*-
import argparse
from tools import *

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMCls to ONNX')
    parser.add_argument('--misid', help='misid', default="lishengxi")
    parser.add_argument('--project', help='project', default="platcv")
    parser.add_argument('--appid', help='server applications id', default="0000")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    params = dict(
        misid=args.misid,
        project=args.project,
        appid=args.appid)

    try:
        print("服务停止参数:{}".format(params))
        print("开始停止efficient serving服务")
        response = stop_efficient_serving(params)
        print("停止结果:{}".format(response))
    except NameError:
        print("服务停止失败")






