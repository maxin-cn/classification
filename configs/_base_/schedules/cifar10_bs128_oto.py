# optimizer
optimizer = dict(type='HSPG', lr=0.1, lmbda=1e-3, momentum=0.0)
optimizer_config = dict(n_p=75, grad_clip=False, type='HSPGOptimizerHook')
# learning policy
lr_config = dict(policy='step', step=[75, 150, 225])
runner = dict(type='EpochBasedRunner', max_epochs=200)