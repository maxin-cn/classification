
_base_ = [
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', 
    '../_base_/default_runtime.py'
]

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='DenseNet', arch='121'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

data = dict(samples_per_gpu=256)

runner = dict(type='EpochBasedRunner', max_epochs=90)