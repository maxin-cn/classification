_base_ = [
    '../_base_/models/efficientnet/tf_efficientnetv2_m.py', '../_base_/datasets/imagenet_bs32_384.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
