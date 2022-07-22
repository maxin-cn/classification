_base_ = [
    '../_base_/models/twins/twins_svt_small.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs256_adamw.py',
    '../_base_/default_runtime.py'
]
