_base_ = [
    '../../_base_/models/resnet18_cifar.py', '../../_base_/datasets/cifar10_bs16.py',
    '../../_base_/schedules/cifar10_bs128.py', '../../_base_/default_runtime.py'
]
# load_from = 'modelzoo/resnet18_top1_90_62.pth'
load_from = '/home/hadoop-basecv/cephfs/data/songshuang/MT-CVZOO/infra-mt-cvzoo-classification/modelzoo/resnet18_top1_90_62.pth'
# pruning settings
compress = True
# compression strategy ['prune', 'quant']
compress_op = 'prune'
compress_not_prune_layer = []
