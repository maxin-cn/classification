# -*- coding: UTF-8 -*-
import json
import sys, os


if __name__ == '__main__':

    version_info = {
        "codebase_version": "0.0.1",
        "codebase_name":"mt-cvzoo-classification"
    }
    
    with open("mtcvzoo", "w") as f:
        json.dump(version_info, f)


    codebase_params = {
        "codebase_name":"mt-cvzoo-classification",
        "codebase_version":"0.0.1",
        "training":{
            "network":"FANet",
            "pretrained_model":"***",
            "task_type":"classification"
         },
        "automl":{
            "algorithm":{},
            "task":{},
            "component":{},
            "architecture":{}
        },
        "compression":{},
        "convertion":{},
        "deployment":{}
    }

    with open("codebase_params.json", "w") as f:
        json.dump(codebase_params, f)
