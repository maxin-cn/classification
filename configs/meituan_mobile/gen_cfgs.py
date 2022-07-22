import jinja2
import os


_dir = 'mmcls/models/backbones/meituan_mobile/net_config'
dest = 'configs/meituan_mobile'
env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.abspath(os.path.dirname(__file__)) + '/templates'))
proto_template = env.get_template('meituan_cv_cifar_template.py')
for filename in os.listdir(_dir):
    if filename.endswith(".config"):
        model = filename[:-7]
        print(model)
        with open(dest+"/%s_cifar.py" % model, 'w') as f:
            url = 'https://s3plus.sankuai.com/v1/mss_9240d97c6bf34ab1b78859c3c2a2a3e4/automl-model-zoo/openmmlab/net_pretrained/' + model +'.pth'
            f.write(proto_template.render(model=model, url=url))