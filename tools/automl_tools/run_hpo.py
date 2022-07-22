from Atum.models.hpo import HPOManageService

if __name__ == '__main__':
    hpo_service = HPOManageService()
    config = dict()
    config['search_space'] = {
        'learning_rate': {'_type': 'uniform', '_value': [0.001, 0.005]},
        'batch_size': {'_type': 'choice', '_value': [32, 64, 128]}
    }
    config['parallel_search'] = 8
    config['gpu_number_per_task'] = 1
    config['optimize_mode'] = 'maximize'
    config['search_time'] = '1h'
    config['task_command'] = "python /mnt/beegfs/ssd_pool/docker/user/hadoop-automl/wengkaiheng/code/infra-mt-cvzoo-classification/tools/automl_tools/train_hpo.py /mnt/beegfs/ssd_pool/docker/user/hadoop-automl/wengkaiheng/code/infra-mt-cvzoo-classification/configs/custom/Res18_cifar.py --hpo"
    hpo_service.update_config(config)
    hpo_service.start()
