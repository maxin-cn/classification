# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='FANet',
        base_width=96,
        layers=[3, 6, 3],
        groups=2,
        stem="deep",
        att_type="slse",
        avd_first=False),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=96*16,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
