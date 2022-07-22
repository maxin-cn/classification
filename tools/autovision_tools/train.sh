#!/bin/bash
# get machine info
cluster_spec=${AFO_ENV_CLUSTER_SPEC//\"/\\\"}
echo "cluster spec is $cluster_spec"
worker_list_command="import json;print(json.loads(\"$cluster_spec\")[\"worker\"])"
echo "worker list command is $worker_list_command"
worker_list=`python -c "$worker_list_command"`
echo "worker list is $worker_list"
worker_strs=(${worker_list//[/ })
worker_strs=${worker_strs[0]}
worker_strs=(${worker_strs//]/ })
master=${worker_strs[0]}
echo "master is $master"
master_strs=(${master//:/ })
master_addr=${master_strs[0]}
master_port=${master_strs[1]}
master_addr=(${master_addr//\'/})
master_port=(${master_port//\'/})
master_port=(${master_port//,/})
echo "master address is $master_addr"
echo "master port is $master_port"
index_command="import json;print(json.loads(\"$cluster_spec\")[\"index\"])"
eval node_rank=`python -c "$index_command"`
echo "node rank is $node_rank"
dist_url="tcp://$master_addr:$master_port"
echo "dist url is $dist_url"
resource_spec=${AFO_RESOURCE_CONFIG//\"/\\\"}
echo "resource spec is $resource_spec"
gpu_num_command="import json;print(json.loads(\"$resource_spec\")[\"worker\"][\"gpu\"])"
echo "gpu num command is $gpu_num_command"
eval gpu_num=`python -c "$gpu_num_command"`
echo "worker list is $gpu_num"
export PATH=/usr/local/anaconda3/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/TensorRT-8.2.0.6/lib:/usr/local/cudnn-8.2.4.15/lib64:/usr/local/anaconda3/lib:${LD_LIBRARY_PATH}

train_path=$(grep -Po 'train_path[" :]+\K[^"]+' /workdir/config.txt)
echo $train_path
pretrained_model_path=$(grep -Po 'pretrained_model_path[" :]+\K[^"]+' /workdir/config.txt)
if [ -n "${pretrained_model_path}" ]; then
    echo "pretrained_model_path is :${pretrained_model_path}"
else
    echo "pretrained_model_path is null"
fi
# install cv-zoo-classification and train
cd infra-mt-cvzoo-classification
rm -rf mmcls.egg-info && pip install -e .
python -m torch.distributed.launch --nproc_per_node=${gpu_num} --node_rank=${node_rank} --master_addr=${master_addr} --master_port=${master_port} tools/train.py /workdir/train_config.py --launcher pytorch --work-dir ${train_path}
export_path="${train_path}/export"
mkdir -p ${export_path}
export_model="${train_path}/latest.onnx"
echo "move ${export_model} to export path: ${export_path}"
mv ${export_model} ${export_path}