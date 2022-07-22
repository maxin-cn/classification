# -*- coding: UTF-8 -*-
import argparse
import hashlib
import json
import sys,os
import traceback
import time
import datetime
import re
import email.utils
if sys.version_info.major == 2:
    from urlparse import urlparse
else:
    from urllib.parse import urlparse
import hmac
import codecs
import base64
import requests
import shutil
import logging
from tools import *

VISION_SERVER_URL = "http://horus.sankuai.com"
VISION_REPORT_VERIFY_CODE_URI = '/mtflow/codebaseapi/report-codebase-params'
VISION_BA_KEY = "mtcvzoo"
VISION_BA_SECRET = "t6bsmwbdfhcd5w4wln578sn0ciimupmr"

class MWSAuth(requests.auth.AuthBase):
    """
    requests's New Forms of Authentication
    http://docs.python-requests.org/en/latest/user/authentication/#new-forms-of-authentication

    美团签名请求规范
    http://wiki.sankuai.com/pages/viewpage.action?pageId=29755412
    """
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret

    def _create_http_date(self, now=None):
        """
        RFC 1123 Date Representation in Python
        http://stackoverflow.com/questions/225086/rfc-1123-date-representation-in-python
        """
        now = now or datetime.datetime.now()
        stamp = time.mktime(now.timetuple())
        return self._formatdate(
            timeval=stamp,
            localtime=False,
            usegmt=True
        )

    @staticmethod
    def _gen_signature(client_secret, method, path, http_date):
        str_to_sign = "%s %s\n%s" % (method, path, http_date)
        signature = base64.b64encode(hmac.new(codecs.encode(client_secret), codecs.encode(str_to_sign), hashlib.sha1).digest())
        return signature

    @staticmethod
    def _formatdate(timeval=None, localtime=False, usegmt=False):
        if timeval is None:
            timeval = time.time()
        if localtime:
            now = time.localtime(timeval)
            if time.daylight and now[-1]:
                offset = time.altzone
            else:
                offset = time.timezone
            hours, minutes = divmod(abs(offset), 3600)
            if offset > 0:
                sign = '-'
            else:
                sign = '+'
            zone = '%s%02d%02d' % (sign, hours, minutes // 60)
        else:
            now = time.gmtime(timeval)
            if usegmt:
                zone = 'GMT'
            else:
                zone = '-0000'
        return '%s %d %s %04d %02d:%02d:%02d %s' % (
            ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][now[6]],
            now[2],
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][now[1] - 1],
            now[0], now[3], now[4], now[5],
            zone)

    @staticmethod
    def _gen_authorization(client_id, signature):
        return "MWS %s:%s" % (client_id, signature.decode(encoding='utf-8'))

    def __call__(self, r):
        http_date = self._create_http_date()
        url_parts = urlparse(r.url)

        signature = self._gen_signature(self.client_secret, r.method, url_parts.path, http_date)
        authorization = self._gen_authorization(self.client_id, signature)

        r.headers['Date'] = http_date
        r.headers['Authorization'] = authorization
        r.headers['Content-Type'] = 'application/json'
        return r

def report_verify_code(report_info):
    retry = 1
    while retry < 3:
        res = None
        try:
            res = requests.post(VISION_SERVER_URL + VISION_REPORT_VERIFY_CODE_URI, data=json.dumps(report_info), auth = MWSAuth(VISION_BA_KEY, VISION_BA_SECRET))
            if res.ok:
                if res.json().get('code') == 0:
                    print("report success")
                    res.close()
                    return True
                else:
                    print("report failed, (%(code)s) %(message)s" % (res.json()))
        finally:
            if res and isinstance(res, requests.models.Response):
                res.close()
        print("retry %d" % retry)
        retry += 1
    if res and isinstance(res, requests.models.Response):
        res.close()
    return False

def extract_model_zoo_info(filename):
    contents, rows = read_text(filename)
    model_zoo_info = dict()
    for i in range(1, rows):
        model_info = contents[i].split(",")
        model_zoo_info[model_info[0]] = {"author": model_info[-2], "group": model_info[-1]}
    print("model_zoo_info:", model_zoo_info)
    return model_zoo_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--codebase_params_file', type=str)
    args = parser.parse_args()
    
    job_id = -1
    submit_type = -1
    vision_info_filename = "vision_info"
    if os.path.exists(vision_info_filename):
        pass
    elif os.path.exists(os.path.join("/workdir", vision_info_filename)):
        vision_info_filename = os.path.join("/workdir", vision_info_filename)
    else:
        print("vision_info not found, exit!")
        sys.exit(1)
    
    with open(vision_info_filename, "r") as f:
        vision_info = json.load(f)
        print("vision_info: {}".format(vision_info))
        if "job_id" in vision_info.keys():
            job_id = vision_info["job_id"]
        else:
            print("job_id not in vision_info, exit!")
            sys.exit(1)
        if "submit_type" in vision_info.keys():
            submit_type = vision_info["submit_type"]
        else:
            print("submit_type not in vision_info, exit!")
            sys.exit(1)

    codebase_params_filename = args.codebase_params_file
    if os.path.exists(codebase_params_filename):
        with open(codebase_params_filename, "r") as f:
            codebase_params = json.load(f)

    model_zoo_info_filename = "resources/model-zoo-info.csv"
    if os.path.exists(model_zoo_info_filename):
        model_zoo_info = extract_model_zoo_info(model_zoo_info_filename)
        if "training" in codebase_params.keys() and \
            "network" in codebase_params["training"].keys() and \
            codebase_params["training"]["network"] in model_zoo_info.keys():
            network = codebase_params["training"]["network"]
            codebase_params["training"]["author"] = model_zoo_info[network]["author"]
            codebase_params["training"]["group"] = model_zoo_info[network]["group"]
    else:
        print("resources/model-zoo-info.csv not found")

    '''
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
    '''
    codebase_params = json.dumps(codebase_params)
    print("codebase_params: {}".format(codebase_params))

    report_info = {
        "submitType": submit_type,
        "jobId": job_id,
        "codebaseParams": codebase_params
    }
    report_verify_code(report_info)



