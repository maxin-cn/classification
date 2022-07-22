_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

# 模型转换参数
model_convert = dict(
    input_shape=[4, 3, 224, 224], 
    checkpoint="latest.pth")

# 模型部署参数
model_deploy = dict(
	preprocess_weight="https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/scripts/preprocess_mmclassification.py",
    weight="latest.onnx",
    setFp16=False,
    setInt8=False,
    appkey="com.sankuai.basecv.serving.autodeploy",
    misid="lishengxi",
    project="platcv",
    cluster="GH",
    queue="root.gh_serving.hadoop-vision.serving")



