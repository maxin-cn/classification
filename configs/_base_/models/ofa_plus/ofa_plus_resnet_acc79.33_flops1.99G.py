# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='OFAPlus',
                  config_name="ofa_plus_resnet_acc79.33_flops1.99G"),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))