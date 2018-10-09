"""
Platform independent utilities
"""

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


# WARNING if you call this funtion, you had better also call retain_session_dir
# as well at the suitable time or else it will get deleted on the next session
# run.
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


