import csv
import fnmatch
import itertools
import json
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
#   output_dir: The output directory
def copy_code(project_dir, output_dir):
    project_dir = os.path.abspath(project_dir)
    project_name = os.path.basename(project_dir)
    path_to_ml_utils = os.path.dirname(os.path.dirname(os.path.abspath(\
            __file__)))
    mlu_name = os.path.basename(path_to_ml_utils)
    repos_dir = pj(output_dir, "repos")
    if pe(repos_dir):
        shutil.rmtree(repos_dir)
    pdir = pj(repos_dir, project_name)
    mludir = pj(repos_dir, mlu_name)
    shutil.copytree(project_dir, pdir, ignore=include_patterns("*.py", "*.txt"))
    shutil.copytree(path_to_ml_utils, mludir,
        ignore=include_patterns("*.py", "*.txt"))

# For running inference, copies the configuration saved to session.log using 
# write_session_config during training.  To be run from program that does 
# inference
# Inputs:
#   cfg: configuration file from inference program.  All and only parameters
#       to be updated should have value None
#   framework: [tensorflow, pytorch] what modeling framework is being used
# Output:
#   Same configuration file, now with new values added.
def copy_config(cfg, framework="pytorch"):
    logfile = pj( cfg["source_session_dir"], "session.log" )
    source_cfg = read_session_config(logfile)
    if framework=="pytorch":
        suffixes = [".pth", ".pt", ".pkl"]
    else:
        raise NotImplementedError()
    models_dir = pj(source_cfg["session_dir"], "models")
    cfg["model_path"] = get_recent_model( models_dir, model_suffixes=suffixes )
    if len( cfg["model_path"] ) == 0:
        raise RuntimeError("Model path not found, looked in %s for files " \
                "ending with %s" % (models_dir, repr(suffixes)))
    for k,v in cfg.items():
        if v is None:
            if k not in source_cfg:
                print("Warning, key %s is missing.  This is okay if running " \
                        "inference on an older dataset" % k)
                cfg[k] = None
            else:
                try:
                    v = int( source_cfg[k] )
                except ValueError:
                    try:
                        v = float( source_cfg[k] )
                    except ValueError:
                        v = source_cfg[k]
                cfg[k] = v
    if len( cfg["output_dir"] ) == 0:
        cfg["output_dir"] = pj(source_cfg["session_dir"], "eval")
    if not pe(cfg["output_dir"]):
        os.makedirs( cfg["output_dir"] )
    print("Configuration:")
    devnull = [print("\t%s: %s" % (k,v)) for k,v in cfg.items()]
    return cfg
    
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

# Inputs
#   x: A PIL image
#   aspect_ratio: aspect ratio of cropped region, width/height
# Output
#   Returns the largest rectangle possible with the specified aspect ratio,
#   cropped from the image and retaining the original center.
def crop_to_ar(x, aspect_ratio):
    o_wd,o_ht = x.size
    o_ar = o_wd/o_ht
    if aspect_ratio > o_ar:
        wd = o_wd
        ht = int(o_wd / aspect_ratio)
        x0 = 0
        y0 = int((o_ht - wd/aspect_ratio) // 2)
    else:
        wd = int(aspect_ratio * o_ht)
        ht = o_ht
        x0 = int((o_wd - aspect_ratio*ht) // 2)
        y0 = 0
    return x.crop((x0, y0, x0+wd, y0+ht))

# Intended for csv files produced by calls to write_training_results.  This 
# creates a file *_reduced.csv with all columns in which there is no 
# variability, removed.  It overwrites any preexisting file of this name.
# Inputs:
#   csv_path: Path to csv file
#   drop_rows (optional): If integer, drop first n rows, default=0; if list of
#       ints, drop these rows.  The first non-header row is row 1.
#   save (optional): List of column names to save regardless of variability
#   remove (optional): List of column names to remove, regardless of variability
# Output:
#   None
def csv_reducer(csv_path, drop_rows=0, save=[], remove=[]):
    csv_path = os.path.abspath(csv_path)
    reader = csv.reader( open(csv_path) )
    header = next(reader)
    rows = []
    if type(drop_rows) == int:
        drop_rows = list(range(drop_rows))
    for i,row in enumerate(reader):
        if i not in drop_rows:
            rows.append(row)
    num_rows = len(rows)
    num_cols = len(header)
    cols = []
    for j in range(num_cols):
        cols.append([])
    for i in range(num_rows):
        for j in range(num_cols):
            cols[j].append( rows[i][j] )
    reduced_header = []
    reduced_cols = []
    for h,col in zip(header,cols):
        pruned_col = [c for c in col if (c is not None and len(c) > 0)]
        if h in save or (h not in remove and len( np.unique(pruned_col) ) > 1):
            reduced_header.append(h)
            reduced_cols.append(col)
    reduced_rows = []
    for i in range(num_rows):
        row = []
        for col in reduced_cols:
            row.append( col[i] )
        reduced_rows.append(row)

    file_stub = os.path.splitext( os.path.basename(csv_path) )[0]
    file_dir = os.path.dirname(csv_path)
    reduced_csv = pj(file_dir, file_stub+"_reduced.csv")
    writer = csv.writer( open(reduced_csv, "w") )
    writer.writerow(reduced_header)
    for row in reduced_rows:
        writer.writerow(row)
    
# Inputs
#   x: A PIL image
#   expansion: If this is an int, the pixel size of the new image.  If it is a
#       float, the proportion increase (e.g, 1.25).
#   fill_color: If the image is RGB, this color will be used in the expansion
# Output
#   Returns a larger image with the original image centered in the original
def expand_square_image(x, expansion, fill_color=(0,0,0)):
    wd,ht = x.size
    max_dim = max(wd, ht)
    if expansion==max_dim or expansion==1.0:
        return x
    if type(expansion)==int:
        if expansion<max_dim:
            raise RuntimeError("The new dimension %d must be larger than the " \
                    "max input image dimension, %d" % (expansion, max_dim))
        new_sz = expansion
    elif type(expansion)==float:
        if expansion<1.0:
            raise RuntimeError("Expansion factor must be >= 1.0, got %f" \
                    % expansion)
        new_sz = int( np.round( expansion * max_dim ) )
    else:
        raise RuntimeError("Unrecognized type for expansion, %s" \
                % type(expansion))
    if x.mode=="L":
        # TODO np.array is very slow here, doesn't seem to be any way to do this
        # without it though.
        orig = np.array( x.getdata() ).reshape(ht,wd)
        x0 = (new_sz - wd) // 2
        y0 = (new_sz - ht) // 2
        x1 = x0 + wd
        y1 = y0 + ht
        img = np.zeros((new_sz, new_sz))
        img[y0:y1, x0:x1] = orig

        c0 = np.expand_dims(img[:, x0], 1)
        img[:, :x0] = c0
        c1 = np.expand_dims(img[:, x1-1], 1)
        img[:, x1:] = c1
        r0 = np.expand_dims(img[y0, :], 0)
        img[:y0, :] = r0
        r1 = np.expand_dims(img[y1-1, :], 0)
        img[y1:, :] = r1

        img = Image.fromarray(img).convert("L")

    elif x.mode=="RGB":
        img = Image.new("RGB", (new_sz,new_sz), fill_color)
        img.paste( x, ((new_sz - wd) // 2, (new_sz - ht) // 2) )

    else:
        raise RuntimeError("Unexpected PIL image mode, %s" % x.mode)
    return img

# Return the path to the directory of the project containing the given file
# Inputs:
#   project_name: Name of project, e.g. "retina"
#   file_path: path to file
# Output:
#   Full path to the project directory containing the file
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
#   models_dir: Directory containing pytorch models saved in the format used by
#       save_model_pop_old
# Output
#   Full path to most recent model
def get_recent_model(models_dir, model_suffixes=[".pkl", ".pt", ".pth"]):
    models = []
    for f in os.listdir(models_dir):
        for ms in model_suffixes:
            if f.endswith(ms):
                models.append(f)
                break
    best_num = -1
    best_path = ""
    for m in models:
        m_ = os.path.splitext(m)[0]
        if int(m_[-4:]) > best_num:
            best_num = int(m_[-4:])
            best_path = pj(models_dir, m)
    if len(best_path) == 0:
        raise RuntimeError("No models found in %s" % models_dir)
    return best_path


# Inputs
#   x: A PIL image
#   fill_color: If the image is RGB, this color will be used in the expansion
# Output
#   Returns the minimal square expansion of the image, preserving the center
#   of the original
def grow_image_to_square(x, fill_color=(0,0,0)):
    wd,ht = x.size
    new_sz = np.max([wd, ht])
    if x.mode=="L":
        # TODO np.array is very slow here, doesn't seem to be any way to do this
        # without it though.
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
    elif x.mode=="RGB":
        img = Image.new("RGB", (new_sz,new_sz), fill_color)
        img.paste( x, ((new_sz - wd) // 2, (new_sz - ht) // 2) )
    else:
        raise RuntimeError("Unexpected PIL image mode, %s" % x.mode)
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
def make_or_get_session_dir(sessions_supdir, model_name="", dataset_name="",
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

# Inputs
#   x: any number > 0
# Output
#   Returns first power of 2 *greater than or equal to* x
def nearest_pow2(x):
    x = int(np.ceil(x))
    p2 = 1
    while x > p2:
        p2 *= 2
    return p2

# Cribbed pretty directly from https://scikit-learn.org/stable/auto_examples/
# model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-
# selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, save_path, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          write_to_console=False,
                          show_plot=False):
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
    if show_plot:
        plt.show()
    else:
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
            if len(val) == 0:
                cfg[key] = []
            else:
                if val[0] == "'":
                    val = val[1:]
                if val[-1] == "'":
                    val = val[:-1]
                if len(val)>1 and val[-1] == "," and val[-2] == "'":
                    val = val[:-2] + ","
                cfg[key] = val
            line = next(fp)
    return cfg

# Walks through all the subdirectories of the supplied path recursively, 
# deleting all files with a given extension and/or in a specific subfolder.
# Inputs:
#   root_dir: Root directory
#   subdir (optional): If supplied, file must be within a subfolder of this name
#   ext (optional): If supplied, only files with this extension will be deleted
# Output:
#   None
def recursive_file_delete(root_dir, subdir=None, ext=None):
    ct = 0
    for root,dirs,files in os.walk(root_dir):
        if subdir is None or subdir in root.split(os.sep):
            del_files = files if ext is None \
                    else [f for f in files if f.endswith(ext)]
            _ = [os.remove(pj(root,f)) for f in del_files]
            ct += len(del_files)
    print("%d files removed from directory %s" % (ct, root_dir))

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
#   tb_writer (optional): tensorboard writer
def write_arguments(session_dir, tb_writer=None):
    s = "python %s" % sys.argv[0]
    for arg in sys.argv[1:]:
        s += " %s" % arg
    with open(pj(session_dir, g_session_log), "a") as fp:
        fp.write("Command line:\n")
        fp.write("%s\n\n" % s)
        if tb_writer is not None:
            tb_writer.add_text("Text", "Command line:", 0)
            tb_writer.add_text("Text", "%s\n" % s, 0)

# Writes the parameters and model to a json file
# Inputs:
#   session_dir: The directory containing the session log.  It will be 
#       created if it doesn't exist.
#   keys (optional): List of keys to write out, others will be ignored
#   model_path (optional): Path to model file
#   output_path optional): path where the json file will be written
def write_config_json(logfile, keys=None, model_path=None,
        output_path=None):
    args_dict = read_session_config(logfile)
    for k in args_dict.keys():
        if args_dict[k] == "True":
            v = True
        elif args_dict[k] == "False":
            v = False
        else:
            try:
                v = int( args_dict[k] )
            except ValueError:
                try:
                    v = float( args_dict[k] )
                except ValueError:
                    v = args_dict[k]
        args_dict[k] = v

    if keys is not None:
        d = OrderedDict()
        for k in keys:
            d[k] = args_dict[k]
        args_dict = d

    if model_path is not None:
        if not pe(model_path):
            print("Warning, supplied model path %s doesn't exist" \
                    % model_path)
        args_dict["model_path"] = model_path
    if output_path is None:
        logfile_dir = os.path.dirname( os.path.abspath(logfile) )
        output_path = pj(logfile_dir, "config.json")
    json.dump(args_dict, open(output_path, "w"), indent=4)
    print("Wrote config json file to %s" % output_path)
    return args_dict

# Appends the input parameters to the session log
# Inputs:
#   cfg: A dict containing all of the parameters.  Must contain an entry
#       "session_dir" with a valid path.
#   output_dir: Output directory, default is cfg["session_dir"]
#   output_file: Output file name, default is g_session_log
#   tb_writer (optional): tensorboard writer
def write_parameters(cfg, output_dir=None, output_file=g_session_log,
        tb_writer=None):
    if output_dir is None:
        output_dir = cfg["session_dir"]
    with open(pj(output_dir, output_file), "a") as fp:
        fp.write("Configuration:\n")
        if tb_writer is not None:
            tb_writer.add_text("Text", "Configuration", 0)
        s = ""
        for k,v in cfg.items():
            if type(v)==list or type(v)==tuple:
                s += "%s: " % k
                for elt in v:
                    s += "%s," % repr(elt)
                s += "\n"
            else:
                s += "%s: %s\n" % (k, repr(v))
        fp.write(s + "\n")
        if tb_writer is not None:
            tb_writer.add_text("Text", s, 0)

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

def xpt_to_csv(xpt_file, output_path=None):
    import xport
    xpt_file = os.path.abspath(xpt_file)
    if output_path is None:
        output_path = os.path.splitext(xpt_file)[0] + ".csv"
    writer = csv.writer( open(output_path, "w") )
    with open(xpt_file, "rb") as fpr:
        reader = xport.Reader(fpr)
        writer.writerow( list(reader.fields) )
        for line in reader:
            writer.writerow(line) 
