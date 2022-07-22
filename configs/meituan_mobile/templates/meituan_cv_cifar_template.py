_base_ = [
    '../_base_/models/meituan_mobile/{{model}}.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(type='MeituanMobile',
                  config_name="{{model}}", 
                  init_cfg=dict(
                        type='Pretrained',
                        checkpoint='{{url}}',
                  )),
    head=dict(num_classes=10)
)

lr_config = dict(policy='step', step=[120, 170])
runner = dict(type='EpochBasedRunner', max_epochs=200)