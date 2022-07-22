# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetOTOImageNet',
        layers=[3, 4, 6, 3],
        ),
    head=dict(
        type='OTOClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))