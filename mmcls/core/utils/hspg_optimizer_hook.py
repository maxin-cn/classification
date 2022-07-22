from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class HSPGOptimizerHook(Hook):

    def __init__(self, n_p=75, grad_clip=False, detect_anomalous_params=False):
        self.n_p = n_p
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params

    # def before_run(self, runner):
    #     pass

    # def after_run(self, runner):
    #     pass

    # def before_epoch(self, runner):
    #     pass

    def after_epoch(self, runner):
        _, _, _, sparsity_group, omega = runner.optimizer.compute_group_sparsity_omega()
        # print('sparsity_group: {}'.format(sparsity_group), 'omega: {}'.format(omega))
        runner.log_buffer.output['sparsity_group'] = sparsity_group
        runner.log_buffer.output['omega'] = omega

    # def before_iter(self, runner):
    #     pass

    def after_iter(self, runner):
        epoch = runner.epoch
        # print('*' * 100 + 'after_iter' + 'epoch:{}'.format(epoch) + '*' * 100)

        runner.optimizer.zero_grad()

        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)

        runner.outputs['loss'].backward()
        # runner.outputs['loss'].backward(retain_graph=True)

        if epoch < self.n_p:
            # print('epoch: {}'.format(iteration), 'in sgd_step')
            runner.optimizer.sgd_step()
        else:
            # print('iter: {}'.format(iteration), 'in half_space_step')
            runner.optimizer.half_space_step()

    def detect_anomalous_parameters(self, loss, runner):#
        logger = runner.logger
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, 'variable'):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in runner.model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')


    # def after_train_iter(self, runner):
    #     runner.optimizer.zero_grad()
    #     if self.detect_anomalous_params:
    #         self.detect_anomalous_parameters(runner.outputs['loss'], runner)
    #     runner.outputs['loss'].backward()

    #     if self.grad_clip is not None:
    #         grad_norm = self.clip_grads(runner.model.parameters())
    #         if grad_norm is not None:
    #             # Add grad norm to the logger
    #             runner.log_buffer.update({'grad_norm': float(grad_norm)},
    #                                      runner.outputs['num_samples'])
    #     runner.optimizer.step()

