_base_ = [
    '../../_base_/models/resnet18.py', '../../_base_/datasets/imagenet_bs32.py',
    '../../_base_/schedules/imagenet_bs256_coslr.py', '../../_base_/default_runtime.py'
]
load_from = 'modelzoo/resnet18-imgnet.pth'
# pruning settings
compress = True
# compression strategy ['prune', 'quant']
compress_op = 'prune'
compress_not_prune_layer = []