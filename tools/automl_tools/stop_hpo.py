from Atum.models.hpo import HPOManageService

if __name__ == '__main__':
    hpo_service = HPOManageService(port=8088)
    hpo_service.stop()