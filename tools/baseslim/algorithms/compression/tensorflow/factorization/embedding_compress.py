# Copyright (c) Meituan.
# author: liujunjie10@meituan.com

from __future__ import absolute_import, division, print_function

import sys
sys.path.append(".")
import os
import time
import numpy as np
from argparse import ArgumentParser
from tensorflow.python import pywrap_tensorflow
from SKDX.algorithms.compression.tensorflow.factorization.matrix_transformer import EmbeddingCompressor

class DefaultArgs:
    def __init__(self):
        self.model_type = "DIN"
        self.M = 8 
        self.K = 8
        self.max_epoches = 200
        self.batch_size = 256
        self.embed_shape = [1000002, 8]
        self.ckpt_name = None
        self.ckpt_path = None
        self.export_name = None

def _check_variable_need_compress(checkpoint_path):
    print(checkpoint_path)
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_compress = {}
    for key in var_to_shape_map:
        if "d/part_" in key:
            var_to_compress[key.split("/d/part")[0]] = var_to_shape_map[key]
    return var_to_compress

def compress_check(ckpt_path=None, ckpt_name=None):
    # check the variable in network need to be compressed
    var_to_compress = _check_variable_need_compress(os.path.join(ckpt_path, ckpt_name))
    _list_var = list(var_to_compress)
    print("==> the following vars are detected to be compressed, please comfirm if you want to compress them ")
    for i in range(len(_list_var)):
        print(i, _list_var[i])
    return _list_var
    
def compress_run(ckpt_path=None, ckpt_name=None, export_name=None, to_compress_vars=None):
    args = DefaultArgs()
    args.ckpt_path = ckpt_path
    args.ckpt_name = ckpt_name
    args.export_name = export_name
    compressor = EmbeddingCompressor(args.M, args.K, args.batch_size, args.embed_shape, args.ckpt_name, args.ckpt_path, args.export_name, to_compress_vars)
    compressor.train(args.max_epoches)
    return "compress finished"
    #elif args.export:
    #    compressor = EmbeddingCompressor(args.M, args.K, args.batch_size, args.embed_shape, args.ckpt_name, args.ckpt_path, args.export_name, _name_each, False)
    #    compressor.export(args.export_name)

if __name__ == '__main__':
    print("load mtcompress success!")

def test():
    ap = ArgumentParser()
    ap.add_argument("--model_type", default="DIN")
    ap.add_argument("--export_name", default="dict")
    ap.add_argument("--ckpt_path", default="./new_ckpt")
    ap.add_argument("--ckpt_name", default="model.ckpt-4585038")
    ap.add_argument("--M", default=8, type=int)
    ap.add_argument("--K", default=8, type=int)
    ap.add_argument("--max_epoches", default=200, type=int)
    ap.add_argument("--batch_size", default=256, type=int)
    ap.add_argument("--embed_shape", default=[1000002, 8], type=int)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    # mkdir the output dir
    export_path = "results"
    if os.path.exists(export_path) == False:
        os.mkdir(export_path) 

    # check the variable in network need to be compressed
    var_to_compress = _check_variable_need_compress(os.path.join(args.ckpt_path, args.ckpt_name))
    _list_var = list(var_to_compress)
    print("==> the following vars are detected to be compressed, please enter the corresponding indexes (separate by comma, eg: 0,1,2) if you want to compress them ")
    for i in range(len(_list_var)):
        print(i, _list_var[i])

    print("waiting for your input: ", end=" ")
    user_input = input()
    _vars = user_input.split(",")
    for i in range(len(_vars)):
        _index_each = int(_vars[i])
        _name_each  = _list_var[_index_each]
        args.embed_shape = var_to_compress[_name_each]

        if args.train:
            compressor = EmbeddingCompressor(args.M, args.K, args.batch_size, args.embed_shape, args.ckpt_name, args.ckpt_path, args.export_name, _name_each)
            compressor.train(args.max_epoches)
        elif args.export:
            compressor = EmbeddingCompressor(args.M, args.K, args.batch_size, args.embed_shape, args.ckpt_name, args.ckpt_path, args.export_name, _name_each, False)
            compressor.export(args.export_name)
