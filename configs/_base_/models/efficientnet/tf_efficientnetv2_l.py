# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='TIMMBackbone',
        model_name='tf_efficientnetv2_l',
        exportable=True
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False,
    ))
