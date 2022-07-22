_base_ = [
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', 
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ShuffleNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

lr_config = dict(policy='step', step=[120, 170])
runner = dict(type='EpochBasedRunner', max_epochs=200)