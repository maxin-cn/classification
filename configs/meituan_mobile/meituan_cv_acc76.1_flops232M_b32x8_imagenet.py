_base_ = [
    '../_base_/models/meituan_mobile/meituan_cv_acc76.1_flops232M.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]