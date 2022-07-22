_base_ = [
    '../_base_/models/meituan_mobile/meituan_cv_acc73.4_flops119M.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(type='MeituanMobile',
                  config_name="meituan_cv_acc73.4_flops119M", 
                  init_cfg=dict(
                        type='Pretrained',
                        checkpoint='https://s3plus.sankuai.com/v1/mss_9240d97c6bf34ab1b78859c3c2a2a3e4/automl-model-zoo/openmmlab/net_pretrained/meituan_cv_acc73.4_flops119M.pth',
                  )),
    head=dict(num_classes=10)
)

lr_config = dict(policy='step', step=[120, 170])
runner = dict(type='EpochBasedRunner', max_epochs=200)