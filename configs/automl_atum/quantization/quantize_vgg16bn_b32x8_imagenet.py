_base_ = [
    '../../_base_/models/vgg16bn.py',
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/schedules/imagenet_bs256.py', '../../_base_/default_runtime.py'
]

model = dict(
    pretrained='https://s3plus.meituan.net/v1/mss_9240d97c6bf34ab1b78859c3c2a2a3e4/automl-model-zoo/openmmlab/model_zoo/vgg16_bn_batch256_imagenet_20210208-7e55cd29.pth',
)
data = dict(
    train=dict(
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/train',),
    val=dict(
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val',
        ann_file='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val.txt',),
    test=dict(
        data_prefix='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val',
        ann_file='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/common/imagenet/val.txt',))

algorithm = dict(
    type='NaiveQuantize',
    task='Quantization',
    QAT=True,
    save_quant_onnx=False,
    w_bits=8,
    a_bits=8,
    quant_level=0,  # 0-per_channel; 1-per_layer
    symmtric_type=0,  # 0-symmetric; 1- asymmetric
    calib_batch=50,
)

# Quantize-Aware Training schedule config
# lr is set for a batch size of 128
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=10)
log_config = dict(interval=100)