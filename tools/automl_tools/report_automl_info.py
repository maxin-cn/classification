import sys
def report_automl_info(config):
    import json
    from tools.deployment.tools import read_json, write_json, runcmd
    codebase_info = read_json("mtcvzoo")
    codebase_info["automl"] = {"algorithm": config.algorithm["type"], # SPOS, FairNAS, AutoSlim, L2
                               "task": config.algorithm["task"], # NAS, Pruning, KD, HPO, Quantization
                               # "component": "mutator", # 'searcher', 'distiller', 'quantizer'
                               # "architecture": config.algorithm["architecture"]["type"],
                              }
    config.algorithm.pop("task")
    print("codebase_info:", codebase_info)
    write_json("automl_info.json", codebase_info)
    binary = "python tools/deployment/report_codebase_info.py"
    codebase_params = "--codebase_params_file automl_info.json"
    command = " ".join((binary, codebase_params))
    if runcmd(command):
        print("report_automl_info success")
    else:
        print("report_automl_info fail")
