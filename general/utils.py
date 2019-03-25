"""
Platform independent utilities
"""

import csv
import fnmatch
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import shutil
import sys

from collections import OrderedDict
from PIL import Image


pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

g_delete_me_txt = "delete_me.txt"
g_session_dir_stub = "session_%02d"
g_session_log = "session.log"


# Copies the code (*.py, *.txt) files from the project repo as well as this
# ml_utils repo into a session_dir subdirectory
# Inputs:
#   project_dir: The directory containing the python code repo
#   session_dir: The training session directory
def copy_code(project_dir, session_dir):
    project_dir = os.path.abspath(project_dir)
    project_name = os.path.basename(project_dir)
    path_to_ml_utils = os.path.dirname(os.path.dirname(os.path.abspath(\
            __file__)))
    mlu_name = os.path.basename(path_to_ml_utils)
    repos_dir = pj(session_dir, "repos")
    if pe(repos_dir):
        shutil.rmtree(repos_dir)
    pdir = pj(repos_dir, project_name)
    mludir = pj(repos_dir, mlu_name)
    shutil.copytree(project_dir, pdir, ignore=include_patterns("*.py",
        "*.txt"))
    shutil.copytree(path_to_ml_utils, mludir,
        ignore=include_patterns("*.py", "*.txt"))

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

def get_project_dir(project_name, file_path):
    old_fp = os.path.abspath(file_path)
    while os.path.basename(os.path.abspath(file_path)) != project_name:
        file_path = os.path.dirname(file_path)
        if file_path == old_fp:
            raise RuntimeError("No project directory %s found in path %s" \
                    % (project_name, file_path))
        old_fp = file_path
    return os.path.abspath(file_path)

# Inputs
#   x: A PIL image
# Output
#   Returns the minimal square expansion of the image, preserving the center
#   of the original
def grow_image_to_square(x, fill_color=(0,0,0)):
    wd,ht = x.size
    new_sz = np.max([wd, ht])
    if x.mode=="L":
        orig = np.array( x.getdata() ).reshape(ht,wd)
        if wd<ht:
            x0 = (new_sz - wd) // 2
            y0 = 0
        else:
            x0 = 0
            y0 = (new_sz - ht) // 2
        x1 = x0 + wd
        y1 = y0 + ht
        img = np.zeros((new_sz, new_sz))
        img[y0:y1, x0:x1] = orig
        if wd<ht:
            m0 = np.expand_dims(img[:, x0], 1)
            img[:, :x0] = m0
            m1 = np.expand_dims(img[:, x1-1], 1)
            img[:, x1:] = m1
        else:
            m0 = np.expand_dims(img[y0, :], 0)
            img[:y0, :] = m0
            m1 = np.expand_dims(img[y1-1, :], 0)
            img[y1:, :] = m1
        img = Image.fromarray(img).convert("L")
    else:
        img = Image.new("RGB", (new_sz,new_sz), fill_color)
        img.paste( x, ((new_sz - wd) // 2, (new_sz - ht) // 2) )
    return img

# Copied directly from https://stackoverflow.com/questions/35155382/copying      # -specific-files-to-a-new-folder-while-maintaining-the-original-subdirect
def include_patterns(*patterns):
    """ Function that can be used as shutil.copytree() ignore parameter that
    determines which files *not* to ignore, the inverse of "normal" usage.

    This is a factory function that creates a function which can be used as a
    callable for copytree()'s ignore argument, *not* ignoring files that match
    any of the glob-style patterns provided.

    ‛patterns’ are a sequence of pattern strings used to identify the files to
    include when copying the directory tree.

    Example usage:

        copytree(src_directory, dst_directory,
                 ignore=include_patterns('*.sldasm', '*.sldprt'))
    """
    def _ignore_patterns(path, all_names):
        # Determine names which match one or more patterns (that shouldn't be
        # ignored).
        keep = (name for pattern in patterns
                        for name in fnmatch.filter(all_names, pattern))
        # Ignore file names which *didn't* match any of the patterns given that
        # aren't directory names.
        dir_names = (name for name in all_names if os.path.isdir(pj(path,
            name)))
        return set(all_names) - set(keep) - set(dir_names)

    return _ignore_patterns

# Inputs
#   sessions_supdir: Directory containing all sessions directories, e.g.
#       ~/Training/lap
#   model_name: Name of model
#   dataset_name: Name of dataset
#   resume_path: Path to session to be resumed, if applicable
# Output
#   Returns path to either a newly-created session directory or one to be
#       resumed, per inputs
def make_or_get_session_dir(sessions_supdir, model_name, dataset_name,
        resume_path=""):
    supdir = pj(sessions_supdir, model_name, dataset_name)
    if not pe(supdir):
        os.makedirs(supdir)
    if len(resume_path)==0:
        session_dir = create_session_dir(supdir)
    else:
        session_dir = os.path.dirname( os.path.dirname(resume_path) )
        if not os.path.basename(session_dir).startswith("session_"):
            raise RuntimeError("Invalid resume path given, %s" % (resume_path))
    return session_dir

# Cribbed pretty directly from https://scikit-learn.org/stable/auto_examples/
# model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-
# selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, save_path, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          write_to_console=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if write_to_console:
        if normalize:
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Reads a session.log file and copies the configuration (as written by 
# write_parameters) into a dict
# Inputs:
#   session_log: path to session.log file
# Output:
#   OrderedDict of configuration parameters
def read_session_config(session_log):
    cfg = OrderedDict()
    with open(session_log) as fp:
        line = next(fp)
        ct = 0
        while line.strip() != "Configuration:":
            line = next(fp)
            ct += 1
            if ct > 1000:
                raise RuntimeError("Configuration not found")
            continue
        line = next(fp).strip()
        while len(line) > 1:
            pos = line.index(":")
            key = line[:pos]
            val = line[pos+1:].strip()
            if val[0] == "'":
                val = val[1:]
            if val[-1] == "'":
                val = val[:-1]
            cfg[key] = val
            line = next(fp)
    return cfg

# This deletes the 'delete_me.txt' file which would otherwise signify that
# this directory should be overwritten on the next training run.  Paired with
# create_session_dir
def retain_session_dir(session_dir):
    if pe(pj(session_dir, g_delete_me_txt)):
        os.remove( pj(session_dir, g_delete_me_txt) )

# Appends the command-line arguments to the session log
# Inputs:
#   session_dir: The directory containing the session log.  It will be 
#       created if it doesn't exist.
def write_arguments(session_dir):
    s = "python %s" % sys.argv[0]
    for arg in sys.argv[1:]:
        s += " %s" % arg
    with open(pj(session_dir, g_session_log), "a") as fp:
        fp.write("Command line:\n")
        fp.write("%s\n\n" % s)

# Appends the input parameters to the session log
# Inputs:
#   cfg: A dict containing all of the parameters.  Must contain an entry
#       "session_dir" with a valid path.
def write_parameters(cfg):
    with open(pj(cfg["session_dir"], g_session_log), "a") as fp:
        fp.write("Configuration:\n")
        for k,v in cfg.items():
            fp.write("%s: %s\n" % (k, repr(v)))

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
    if any([s not in results_keys for s in ["loss"]]):
        raise RuntimeError("Argument results_dict must have standard keys: " \
                "loss")
    path = pj(project_dir, "%s_results.csv" % (trainer_name))
    data = []
    prev_dict = OrderedDict()
    if pe(path):
        with open(path) as fp:
            reader = csv.reader(fp)
            header = next(reader)
            for row in reader:
                if len(row)>0:
                    data.append(row)
        for i,h in enumerate(header):
            prev_dict[h] = []
            for row in data:
                prev_dict[h].append( row[i] )
    N = len(data)

    cfg_keys = cfg.keys()
    prev_keys = prev_dict.keys()
    new_dict = OrderedDict()

    for pk in prev_keys:
        if pk in cfg_keys:
            break
        new_dict[pk] = prev_dict[pk]
        if pk in results_keys:
            new_dict[pk].append( results_dict[pk] )
        else:
            new_dict[pk] = prev_dict[pk]
            new_dict[pk].append("")
    for rk in results_keys:
        if rk not in new_dict.keys():
            new_dict[rk] = [""] * N
            new_dict[rk].append( results_dict[rk] )

    for pk in prev_keys:
        new_dict[pk] = prev_dict[pk]
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

