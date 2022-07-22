_base_ = [
    '../_base_/models/fanet/fanet_b2.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
	backbone=dict(
		init_cfg=dict(
			type='Pretrained',
			checkpoint='https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/models/mtvision/fanet_b2.pth', 
			# prefix='backbone',
			)),
	head=dict(num_classes=1000),
)

compress=False
