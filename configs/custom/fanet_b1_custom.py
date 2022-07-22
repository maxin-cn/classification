_base_ = [
    '../_base_/models/fanet/fanet_b1.py',
    # '../_base_/datasets/custom_bs64_autoaug.py',
    '../_base_/datasets/custom_bs64_randaug.py',
    '../_base_/default_runtime.py'
]

# Model config
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/models/mtvision/fanet_b1.pth',
    )),
    head = dict(
        num_classes=14,
        # type='MultiLabelLinearClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_soft=True)
    ),
    train_cfg=dict(
        augments=dict(type='BatchCutMix', alpha=1.0, num_classes=14, prob=1.0)
    )
)

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        data_prefix='data/qinzi/train',
        ann_file='data/qinzi/meta/train.txt',
    ),
    val=dict(
        data_prefix='data/qinzi/val',
        ann_file='data/qinzi/meta/val.txt',
    ),
    test=dict(
        data_prefix='data/qinzi/val',
        ann_file='data/qinzi/meta/val.txt',
    )
)

paramwise_cfg = dict(norm_decay_mult=0.0, bias_decay_mult=0.0,)
optimizer = dict(type='AdamW', lr=5e-4, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999), paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-3,
    warmup_by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=65)
