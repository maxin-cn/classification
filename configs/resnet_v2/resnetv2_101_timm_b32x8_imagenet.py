_base_ = [
    '../_base_/models/resnetv2/resnetv2_101_timm.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
