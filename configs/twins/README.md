
# Twins 


## Generate onnx

```shell
for model in 'small' 'large' 'base';do 
    for tag in 'twins_pcpvt' 'twins_svt';do
        python tools/deployment/pytorch2onnx.py configs/twins/${tag}_${model}_b64x8_imagenet.py --output-file onnx_export/${tag}_${model}.onnx --shape 4 3 224 224
    done
done
```
    

## Fix Onnx for deployment

```shell
python tools/deployment/fix_twins_onnx.py 
```

## ES Deploy

```shell
your_app_key='com.sankuai.automl.profile'
your_mis_id='zhangbo97'

for model in 'small' 'base' 'large';do
    python tools/deployment/model_deploy.py \
        --preprocess-weight https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/scripts/preprocess_mmclassification.py \
        --model-weight onnx_export/twins_svt_${model}.onnx \
        --fp16 \
        --appkey ${your_app_key} \
        --misid ${your_mis_id} \
        --cluster GH \
        --queue root.gh_serving.hadoop-vision.serving \
        --network twins_svt_${model} \
        --custom-input 
done
```