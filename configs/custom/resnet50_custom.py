_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/custom_bs64_autoaug.py',
    # '../_base_/datasets/custom_bs64_randaug.py',
    '../_base_/default_runtime.py'
]

# Model config
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/models/pytorch/zhangjinjin/models/scl/resnet50_scl_hard_sample_mining.pth.tar',
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
    workers_per_gpu=8,
    train=dict(
        data_prefix='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/zhangjinjin/data/image_tags/qinzi/train',
        ann_file='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/zhangjinjin/data/image_tags/qinzi/meta/train.txt',
    ),
    val=dict(
        data_prefix='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/zhangjinjin/data/image_tags/qinzi/val',
        ann_file='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/zhangjinjin/data/image_tags/qinzi/meta/val.txt',
    ),
    test=dict(
        data_prefix='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/zhangjinjin/data/image_tags/qinzi/val',
        ann_file='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/zhangjinjin/data/image_tags/qinzi/meta/val.txt',
    )
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.1,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)
