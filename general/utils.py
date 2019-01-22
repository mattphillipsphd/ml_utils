"""
Platform independent utilities
"""

import csv
import numpy as np
import os
import pathlib
import shutil

from collections import OrderedDict

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

g_delete_me_txt = "delete_me.txt"
g_session_dir_stub = "session_%02d"
g_session_log = "session.log"


# WARNING if you call this funtion, you had better also call retain_session_dir
# as well at the suitable time or else the session that was created  will get 
# deleted on the next session run.
def create_session_dir(output_supdir, dir_stub=g_session_dir_stub):
    stub = pj(output_supdir, dir_stub)
    ct = 0
    while pe(stub % (ct)):
        if pe(pj(stub % (ct), g_delete_me_txt)):
            shutil.rmtree(stub % (ct))
            break
        ct += 1
    os.makedirs(stub % (ct))
    pathlib.Path( pj(stub % (ct), g_delete_me_txt) ).touch()
    return stub % (ct)

def retain_session_dir(session_dir):
    if pe(pj(session_dir, g_delete_me_txt)):
        os.remove( pj(session_dir, g_delete_me_txt) )

# Inputs:
#   results_dict: A dictionary of core training metrics, see code for required
#       elements
#   cfg: A dictionary made from the entirety of input parameters to the trainer
#   project_dir: Directory of the project, above model/dataset directories
#   trainer_name: Name of the trainer script (or trainer class as applicable)
def write_training_results(results_dict, cfg, project_dir, trainer_name):
    if not pe(project_dir):
        raise RuntimeError("project_dir %s should already exist"%(project_dir))
    results_keys = results_dict.keys()
    if any([s not in results_keys for s in ["model", "loss", "accuracy",
        "dataset"]]):
        raise RuntimeError("Argument results_dict must have standard keys: " \
                "model, loss, accuracy, dataset keys")
    path = pj(project_dir, "%s_results.csv" % (trainer_name))
    data = []
    prev_dict = OrderedDict()
    if pe(path):
        with open(path) as fp:
            reader = csv.reader(fp)
            header = next(reader)
            for row in reader:
                data.append(row)
        for i,h in enumerate(header):
            prev_dict[h] = []
            for row in data:
                prev_dict[h].append( row[i] )
    N = len(data)

    prev_keys = prev_dict.keys()
    new_dict = OrderedDict()
    for pk in prev_keys:
        new_dict[pk] = prev_dict[pk]
        if pk in results_keys:
            new_dict[pk].append( results_dict[pk] )
        else:
            break
    for rk in results_keys:
        if rk not in new_dict.keys():
            new_dict[rk] = [""] * N
            new_dict[rk].append( results_dict[rk] )
    cfg_keys = cfg.keys()
    for pk in prev_keys:
        if pk in results_keys:
            continue
        if pk in prev_keys:
            new_dict[pk] = prev_dict[pk]
        else:
            new_dict[pk] = [""] * N
        if pk in cfg_keys:
            new_dict[pk].append( cfg[pk] )
        else:
            new_dict[pk].append("")
    for ck in cfg_keys:
        if ck not in new_dict.keys():
            new_dict[ck] = [""] * N
            new_dict[ck].append( cfg[ck] )

    with open(path, "w") as fp:
        writer = csv.writer(fp)
        header = list(new_dict.keys())
        writer.writerow(header)
        for i in range(N+1):
            row = []
            for h in header:
                row.append( new_dict[h][i] )
            writer.writerow(row)

    print("Wrote training results to %s" % path)

