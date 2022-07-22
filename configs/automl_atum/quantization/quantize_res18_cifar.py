_base_ = [
    '../../_base_/datasets/cifar10_bs16.py',
    '../../_base_/schedules/cifar10_bs128.py',
    '../../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/cifar',
        ),
    val=dict(
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/cifar',),
    test=dict(
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/cifar'))

norm_cfg = dict(type='BN')
model = dict(
    type='mmcls.ImageClassifier',
    init_cfg=dict(
            type='Pretrained',
            checkpoint='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/lee/Codebase/mmcls_res18_cifar.pth',
            # checkpoint='https://s3plus.sankuai.com/v1/mss_9240d97c6bf34ab1b78859c3c2a2a3e4/automl-model-zoo/mmcls_res18_cifar.pth',
        ),
    backbone=dict(
        type='ResNet', depth=18),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=10,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5),
    ),
)

algorithm = dict(
    type='NaiveQuantize',
    task='Quantization',
    QAT=False,
    save_quant_onnx=False,
    w_bits=8,
    a_bits=8,
    quant_level=1,      # 0-per_channel; 1-per_layer
    symmtric_type=1,     # 0-symmetric; 1- asymmetric
    calib_batch=50,
)

# Quantize-Aware Training schedule config
# lr is set for a batch size of 128
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=5)
log_config = dict(interval=100)