# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier

# TODO import `wrap_fp16_model` from mmcv and delete them from mmcls
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--pretrained', help='pretrained', action='store_true')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)


    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    from mmcls.models.backbones.timm_backbone import _my_load_checkpoint, TIMMBackbone
    if type(model.backbone) is TIMMBackbone and args.pretrained:
        checkpoint = _my_load_checkpoint(model, args.checkpoint, map_location='cpu', prefix='timm_model.', load_head=True)
    else:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.device == 'cpu':
        model = model.cpu()
        prune_model, _ = model.backbone.prune()
    else:
        model = MMDataParallel(model, device_ids=[0])
        prune_model, _ = model.module.backbone.prune()

    model_save_path = args.checkpoint.split('.')[0] + '_pruned.' + args.checkpoint.split('.')[1]
    torch.save(prune_model, model_save_path)
    print('Pruned Model Save at: {}'.format(model_save_path))
    print('Pruning Model Sucessfully!')

if __name__ == '__main__':
    main()
