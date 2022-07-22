_base_ = [
    '../_base_/models/ofa_plus/ofa_plus_resnet_acc80.25_flops2.99G.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]

load_from = 'https://s3plus.sankuai.com/v1/mss_9240d97c6bf34ab1b78859c3c2a2a3e4/automl-model-zoo/ofa_plus/ofa_plus_resnet_acc80.25_flops2.99G.pth'
