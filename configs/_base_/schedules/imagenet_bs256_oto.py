# optimizer
optimizer = dict(type='HSPG', lr=0.1, lmbda=1e-3, momentum=0.9)
optimizer_config = dict(n_p=35, grad_clip=False, type='HSPGOptimizerHook')
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)