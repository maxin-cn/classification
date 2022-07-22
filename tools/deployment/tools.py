# -*- coding: UTF-8 -*-
import sys, os
import subprocess
import requests
import numpy as np
import cv2
import base64
import json
import hashlib
import hmac
import codecs
from mssapi.s3.connection import S3Connection
import time
import datetime


def file2binary(imagepath):
    with open(imagepath, 'rb') as fin:
        img = fin.read()
    return img

def file2base64(imagepath):
    with open(imagepath, 'rb') as fin:
        img = fin.read()
    img_base64 = base64.b64encode(img)
    return img_base64

def binary2numpy(img):
    img = np.fromstring(img, np.uint8)
    img = cv2.imdecode(img, 1)
    return img

def base642numpy(img_base64):
    img = base64.b64decode(img_base64)
    img = np.fromstring(img, np.uint8)
    img = cv2.imdecode(img, 1)
    return img

def file2string(imagepath):
    img_base64 = file2base64(imagepath) #<class 'bytes'>
    img_base64 = img_base64.decode('UTF-8') #<class 'str'>
    img_json = {"img": img_base64} 
    img_str = json.dumps(img_json, ensure_ascii=False)
    return img_str

def string2numpy(img_str):
    img_json = json.loads(img_str)
    img_base64 = img_json["img"]
    img = base642numpy(img_base64)
    return img

def list2string(text_recs):
    if len(text_recs) == 0:
        text_recs = []
    for i in range(len(text_recs)):
        text_recs[i] = text_recs[i].tolist()
    text_recs_str = json.dumps(text_recs)
    return text_recs_str

def read_json(json_path):
    json_info = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_info = json.load(f)
    else:
        print("{} not found, exit!".format(json_path))
    return json_info

def write_json(json_path, json_info):
    with open(json_path, "w") as f:
        json.dump(json_info, f)

def read_text(text_path):
    file = open(text_path, "r")
    contents = file.readlines()
    rows = len(contents)
    for i in range(rows):
        contents[i] = contents[i].strip()
    file.close()
    return contents, rows

def runcmd(command):
    ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, encoding="utf-8", timeout=900)
    if ret.returncode == 0:
        print("success:",ret)
        return True
    else:
        print("error:",ret)
        return False

def upload_file(remote_filename, local_filename):
    contents, _ = read_text("./tools/deployment/s3info.txt")
    conn = S3Connection(
                aws_access_key_id = contents[0],
                aws_secret_access_key = contents[1],
                is_secure = False,
                host = contents[2])
    bucket = conn.get_bucket("basecv-model")
    key = bucket.new_key(remote_filename)
    key.set_contents_from_filename(local_filename)
    url = key.generate_url(expires_in = 3600*24*30*12*10, force_http = False)
    return url

def report_deployment_info(model_infos):
    codebase_info = read_json("mtcvzoo")
    codebase_info["deployment"] = model_infos
    print("codebase_info:", codebase_info)
    write_json("deployment_info.json", codebase_info)
    binary = "python tools/deployment/report_codebase_info.py"
    codebase_params = "--codebase_params_file deployment_info.json"
    command = " ".join((binary, codebase_params))
    if runcmd(command):
        print("report_deployment_info success")
    else:
        print("report_deployment_info fail")

def submit_efficient_serving(model_infos_user, submit_infos_user):
    model_infos = {
       "modelInfos":[
           {
               "training_framework":"onnx",
               "vision_task_type":"classification",
               "preprocess_type":"opencv",
               "preprocess_method":0,
               "preprocess_weight":"",
               "weight":"",
               "setFp16":False,
               "setInt8":False,
               "int8Mode": "Simple",
               "dynamicRangeFileName": "",
               "device_type": "gpu",
               "version": 1,
               "model_index": 0,
               "pre_model_index": [-1],
               "post_model_index": [1],
               "top_k": 0,
               "graph": "none",
               "label": "",
               "network": "",
               "hadoop_name": ""
           }
       ]
    }
    if os.path.exists(model_infos_user["preprocess_weight"]):
        remote_filename = "MT_CVZoo/preprocess_" + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + ".py"
        model_infos_user["preprocess_weight"] = upload_file(remote_filename, model_infos_user["preprocess_weight"])
    elif model_infos_user["preprocess_weight"].find("http") == -1:
        print("not found {}, failed to submit the deployment job".format(model_infos_user["preprocess_weight"]))
        return
    if os.path.exists(model_infos_user["weight"]):
        remote_filename = "MT_CVZoo/model_" + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + ".onnx"
        model_infos_user["weight"] = upload_file(remote_filename, model_infos_user["weight"])
    elif model_infos_user["weight"].find("http") == -1:
        print("not found {}, failed to submit the deployment job".format(model_infos_user["weight"]))
        return
    if model_infos_user["setInt8"] and len(model_infos_user["dynamicRangeFileName"]) > 0:
        if os.path.exists(model_infos_user["dynamicRangeFileName"]):
            remote_filename = "MT_CVZoo/calibration_" + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + ".table"
            model_infos_user["dynamicRangeFileName"] = upload_file(remote_filename, model_infos_user["dynamicRangeFileName"])
        elif model_infos_user["dynamicRangeFileName"].find("http") == -1:
            print("not found {}, failed to submit the deployment job".format(model_infos_user["dynamicRangeFileName"]))
            return
        model_infos["modelInfos"][0]["setInt8"] = model_infos_user["setInt8"]
        model_infos["modelInfos"][0]["int8Mode"] = model_infos_user["int8Mode"]
        model_infos["modelInfos"][0]["dynamicRangeFileName"] = model_infos_user["dynamicRangeFileName"]

    if model_infos_user["custom_input"] and len(model_infos_user["input_shape"]) == 4:
        b, c, h, w = model_infos_user["input_shape"]
        input_fields = {
            "input_shape_b" : b,
            "input_shape_c" : c,
            "input_shape_h" : h,
            "input_shape_w" : w,
            "input_shape_type": model_infos_user["input_shape_type"],
            "output_shape": model_infos_user["output_shape"],
            "input_names": [
                "input"
            ],
            "output_names": [
                "output"
            ]
        }
        model_infos["modelInfos"][0].update(input_fields)
    else:
        print("Either custom input not specified or length of input shape is not 4, e.g. (NCHW), use default instead")

    model_infos["modelInfos"][0]["preprocess_weight"] = model_infos_user["preprocess_weight"]
    model_infos["modelInfos"][0]["weight"] = model_infos_user["weight"]
    model_infos["modelInfos"][0]["setFp16"] = model_infos_user["setFp16"]
    model_infos["modelInfos"][0]["device_type"] = model_infos_user["device_type"]
    model_infos["modelInfos"][0]["network"] = model_infos_user["network"]
    model_infos["modelInfos"][0]["hadoop_name"] = model_infos_user["hadoop_name"]

    submit_infos = {
        ########## 服务相关 ############
        "appkey": "", # string，必填，提供tfserving服务的appkey
        "wxProject": "", # 必填。服务所属项目组，用于启停服务和查询的权限校验
        "creator": "",
        ########## 模板相关 ############ 
        "gitArgs":{
            "git":"ssh://git@git.sankuai.com/~lishengxi/efficient_serving4.1.git", # 代码仓库地址，必填
            "branch":"master",                    # 代码分支，非必填,默认为master
            "commit":"",                    # 代码分支，非必填
            "userCodeDir":""                # 代码子目录，默认为 /
        },                                  
        ########## 任务相关 ############
        "servingJobArgs":{
            "servingMode": "thrift",              # 必填。服务方式 thrift或restful
            "jobType": "general",                 # 必填。服务类型 general或python
            "serverImage":"registryonline-hulk.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/hadoop-basecv_efficient_serving4.0_210817_stage3-5770fb93", # 服务镜像，必填
            "httpStartCheckUrl":"",               # 服务心跳check路径，服务方式为restful是需要填
            "customedFile":"run_gpu.sh",          # 必填 自定义服务的脚本文件
            "convertModel":True,                  # 必填，是否开启视觉模型转换
            "convertParam":json.dumps(model_infos),         # efficient serving配置参数，当convertModel为ture是需要填写
            "codeRunDir":"",                      # 代码/包路径，非必填，默认为 ./当前目录
            "extra":""                            # 其他参数，非必填
        },
        ########## 资源相关 ############ #
        "resourceArgs":{
            "hdfsCluster": "",      # 运行集群，必填，下方接口获取
            "queue":"",                             # 运行队列
            "workers": 1,                           # 必填。启动的worker机器个数
            "workerMemory": 12288,                  # 必填。每个worker申请的内存大小，单位为M
            "workerVcore": 6,                       # 必填。每个worker申请的vcores
            "workerGpu": 1,                         # 必填。每个worker申请的gpu
            "gcores":"gcores16g",                   # 非必填，显存类型 gcores12g gcores8g ...
            "workerExtra": "-Dafo.role.worker.gpu_driver_version=450.51.06"# 选填。除资源配置和基本信息外，提交作业的其他参数，如设置直连端口、超时时间等，具体参数形式和afo一致，# 额外需要加-D参数（--files=不需要加-D），多个-D参数之间以空格隔开，如-Dkey1=value1 -Dkey2=value2
        }
    }
    submit_infos["appkey"] = submit_infos_user["appkey"]
    submit_infos["wxProject"] = submit_infos_user["project"]
    submit_infos["creator"] = submit_infos_user["misid"]
    submit_infos["resourceArgs"]["hdfsCluster"] = submit_infos_user["cluster"]
    submit_infos["resourceArgs"]["queue"] = submit_infos_user["queue"]
    
    def buildHeader(misid, project):
       gmtTime = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
       request_uri = '/mlapi/tf-serving/custom/startup'
       string2Sign = 'POST %s\n%s' % (request_uri, gmtTime)
       secretKey = 'uYM3oSJpOXkPit0VoGXxM5sNeXVhrug4'
       secretId = 'platcvautovision'
       sign = hmac.new(codecs.encode(secretKey), codecs.encode(string2Sign), hashlib.sha1).digest()
       signature = str(base64.b64encode(sign), 'utf-8').replace("\n", '')
       header = {
           'Date': gmtTime,
           'Authorization': 'MWS' + ' ' + secretId + ':' + signature,
           'Content-Type': 'application/json;charset=UTF-8',
           'AIFree-User': misid,
           'AIFree-WXProject': project
       }
       return header
    
    uri = 'https://mlp.sankuai.com'
    header = buildHeader(submit_infos_user["misid"], submit_infos_user["project"])

    request_uri = '/mlapi/tf-serving/custom/startup'
    url = uri + request_uri
    response = requests.post(url, headers=header, data=json.dumps(submit_infos))
    #print(response.text)
    report_deployment_info(model_infos)
    return response.text

def stop_efficient_serving(submit_infos_user):
    def buildHeader(misid, project, request_uri):
       gmtTime = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
       string2Sign = 'POST %s\n%s' % (request_uri, gmtTime)
       secretKey = 'uYM3oSJpOXkPit0VoGXxM5sNeXVhrug4'
       secretId = 'platcvautovision'
       sign = hmac.new(codecs.encode(secretKey), codecs.encode(string2Sign), hashlib.sha1).digest()
       signature = str(base64.b64encode(sign), 'utf-8').replace("\n", '')
       header = {
           'Date': gmtTime,
           'Authorization': 'MWS' + ' ' + secretId + ':' + signature,
           'Content-Type': 'application/json;charset=UTF-8',
           'AIFree-User': misid,
           'AIFree-WXProject': project
       }
       return header
    
    uri = 'https://mlp.sankuai.com'
    request_uri = '/mlapi/tf-serving/custom/shutdown/' + submit_infos_user["appid"]
    url = uri + request_uri
    header = buildHeader(submit_infos_user["misid"], submit_infos_user["project"], request_uri)
    response = requests.post(url, headers=header, data="")
    return response.text

if __name__ == "__main__":
    '''
    model_infos_user = dict(
        preprocess_weight="https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/scripts/preprocess_mmclassification.py",
        weight="https://s3plus.sankuai.com/v1/mss_8cb9a34d9587426fbf4d3f42b8c31c86/basecv-model/models/pytorch/resnet50/resnet50-19c8e357-bs4.onnx",
        setFp16=False,
        setInt8=False)
    '''
    '''
    model_infos_user = dict(
        preprocess_weight="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-platcv/lishengxi/model/mmclassification/mt-cvzoo-classification/resnet50/preprocess_mmclassification.py",
        weight="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-platcv/lishengxi/model/mmclassification/mt-cvzoo-classification/resnet50/latest.onnx",
        setFp16=False,
        setInt8=False)
    submit_infos_user = dict(
        appkey="com.sankuai.basecv.serving.autodeploy",
        cluster="GH",
        queue="root.gh_serving.hadoop-vision.serving")
    submit_efficient_serving(model_infos_user, submit_infos_user)
    '''
    '''
    json_info = read_json("json_path")
    print("json_info:", json_info)
    '''
    '''
    model_infos = {
       "modelInfos":[
           {
               "training_framework":"onnx",
               "vision_task_type":"classification",
               "preprocess_type":"opencv",
               "preprocess_method":0,
               "preprocess_weight":"",
               "weight":"",
               "setFp16":False,
               "setInt8":False,
               "int8Mode": "Simple",
               "dynamicRangeFileName": ""
           }
       ]
    }
    report_deployment_info(model_infos)
    '''
    pass





