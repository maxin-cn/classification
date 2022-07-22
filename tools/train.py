# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmcls import __version__
from mmcls.apis import set_random_seed, train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import collect_env, get_root_logger
from Atum.core import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument('--pruned_model', help='pruned model for finetune')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def report_compress_info(config):
    import json
    from tools.deployment.tools import read_json, write_json, runcmd
    codebase_info = read_json("mtcvzoo")
    compress_set = None
    if config.optimizer["type"] == 'HSPG':
        compress_set = 'baseSlim HSPG ' + str(config.optimizer["lmbda"])
    codebase_info["training"] = {
                                    "network": config.model["backbone"]["type"],
                                    "compress_set": compress_set,
                                    "task_type":"classification"
                                }
    print("codebase_info:", codebase_info)
    write_json("pretrain_info.json", codebase_info)
    binary = "python tools/deployment/report_codebase_info.py"
    codebase_params = "--codebase_params_file pretrain_info.json"
    command = " ".join((binary, codebase_params))
    if runcmd(command):
        print("report_compress_info success")
    else:
        print("report_compress_info fail")


def report_pretrain_info(config):
    import json
    from tools.deployment.tools import read_json, write_json, runcmd
    codebase_info = read_json("mtcvzoo")
    checkpoint_path = None
    if "init_cfg" in config.model["backbone"].keys() and "checkpoint" in config.model["backbone"]["init_cfg"].keys():
        checkpoint_path = config.model["backbone"]["init_cfg"]["checkpoint"]
    codebase_info["training"] = {
                                    "network": config.model["backbone"]["type"],
                                    "pretrained_model": checkpoint_path,
                                    "task_type":"classification"
                                }
    print("codebase_info:", codebase_info)
    write_json("pretrain_info.json", codebase_info)
    binary = "python tools/deployment/report_codebase_info.py"
    codebase_params = "--codebase_params_file pretrain_info.json"
    command = " ".join((binary, codebase_params))
    if runcmd(command):
        print("report_pretrain_info success")
    else:
        print("report_pretrain_info fail")

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    model.init_weights()

    if args.pruned_model:
        model = torch.load(args.pruned_model)
        print('Load Pruned Model: {}'.format(args.pruned_model))

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmcls version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    
    try:
        logger.info(f'start to report pretrain info')
        if torch.cuda.current_device() == 0:
            report_pretrain_info(cfg)
            report_compress_info(cfg)
        else:
            pass
    except:
        logger.info(f'skip report pretrain info')

    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        device='cpu' if args.device == 'cpu' else 'cuda',
        meta=meta)
    
    def model_convert(args, cfg):
        import time
        time.sleep(2)  # wait for save pytorch model finish like mmcv do that
        binary = "python tools/deployment/pytorch2onnx.py"
        config = str(args.config)
        srcfile = "--checkpoint " + \
            os.path.join(cfg.work_dir, cfg.model_convert["checkpoint"])
        dstfile = "--output-file " + \
            os.path.join(cfg.work_dir, cfg.model_convert["checkpoint"].replace(".pth", ".onnx"))
        input_shape = "--shape " + \
            str(cfg.model_convert["input_shape"][0]) + " " + \
            str(cfg.model_convert["input_shape"][1]) + " " + \
            str(cfg.model_convert["input_shape"][2]) + " " + \
            str(cfg.model_convert["input_shape"][3])
        command = " ".join((binary, config, srcfile, dstfile, input_shape))
        from tools.deployment.tools import runcmd
        if runcmd(command):
            logger.info(f'convert pytorch model to onnx success')
        else:
            logger.info(f'convert pytorch model to onnx fail')
    
    try:
        if cfg.get('model_convert') is None:
            logger.info(f'skip model convert')
        else:
            logger.info(f'model_convert: {cfg.model_convert}')
            logger.info(f'start convert pytorch model to onnx')
            if torch.cuda.current_device() == 0:
                model_convert(args, cfg)
            else:
                pass  # convert model only once
    except NameError:
        logger.info(f'skip model convert')

    def model_deploy(args, cfg):
        import time
        time.sleep(2)  # wait for convert onnx model finish like mmcv do that
        from tools.deployment.tools import submit_efficient_serving
        cfg.model_deploy["weight"] = os.path.join(cfg.work_dir, cfg.model_deploy["weight"])
        response = submit_efficient_serving(cfg.model_deploy, cfg.model_deploy)
        logger.info(response)
    
    try:
        if cfg.get('model_deploy') is None:
            logger.info(f'skip model deploy')
        else:
            logger.info(f'model_deploy: {cfg.model_deploy}')
            logger.info(f'start deploy model use efficient serving')
            if torch.cuda.current_device() == 0:
                model_deploy(args, cfg)
            else:
                pass  # deploy model only once
    except NameError:
        logger.info(f'skip model deploy')
    


if __name__ == '__main__':
    main()
