# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MeituanMobile',
                  config_name="meituan_cv_acc79.7_flops581M"),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))