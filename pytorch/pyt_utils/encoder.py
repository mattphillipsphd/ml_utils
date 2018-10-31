"""
Generates features (encodings) from a model and a dataset
"""

import logging
import numpy as np
import os
import shutil
import torch

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


# Inputs:
#   model: Subclass of nn.Module, must have a method get_features which returns
#   features (same as output for non-autoencoder models).
def compute_features(model, data_loader, gpu_device=0, make_chip_list=True):
    logging.debug("compute_features")
    logging.info("Generating features ...")
    features = []
    chip_list = []
    if torch.cuda.is_available():
        model = model.cuda(gpu_device)
    if make_chip_list:
        data_loader.dataset.set_return_names(True)
        for inputs,_,name in data_loader:
            inputs = torch.autograd.Variable(inputs.cuda(gpu_device))
            outputs = model.get_features(inputs)
            features.append( outputs.cpu().data.numpy() )
            chip_list.append(name)
        data_loader.dataset.set_return_names(False)
        chip_list = np.squeeze( np.concatenate(chip_list) )
    else:
        for inputs,_ in data_loader:
            inputs = torch.autograd.Variable(inputs.cuda(gpu_device))
            outputs = model.get_features(inputs)
            features.append( outputs.cpu().data.numpy() )
    model = model.cpu()
    features = np.concatenate(features)
    logging.info("... Done, %d features generated." % (len(features)))
    features = np.squeeze( features )
    if len(features.shape) > 2:
        raise RuntimeError("Feature matrix has too many dimensions:",
            features.shape)
    return features,chip_list

def compute_labels(model, data_loader, gpu_device=0, make_chip_list=True):
    logging.debug("compute_labels")
    logging.info("Generating labels ...")
    labels = []
    chip_list = []
    if torch.cuda.is_available():
        model = model.cuda(gpu_device)
    if make_chip_list:
        data_loader.dataset.set_return_names(True)
        for inputs,_,name in data_loader:
            inputs = torch.autograd.Variable(inputs.cuda(gpu_device))
            outputs = model(inputs)
            labels.append( outputs.cpu().data.numpy() )
            chip_list.append(name)
        data_loader.dataset.set_return_names(False)
        chip_list = np.squeeze( np.concatenate(chip_list) )
    else:
        for inputs,_ in data_loader:
            inputs = torch.autograd.Variable(inputs.cuda(gpu_device))
            outputs = model(inputs)
            labels.append( outputs.cpu().data.numpy() )
    model = model.cpu()
    labels = np.concatenate(labels)
    logging.info("... Done, %d labels generated." % (len(labels)))
    labels = np.squeeze(labels)
    if len(labels.shape) > 2:
        raise RuntimeError("Labels matrix has too many dimensions:",
            labels.shape)
    return labels,chip_list

# Concatenate two encoding directories into a new directory
def concatenate(enc_dir1, enc_dir2, output_dir):
    def _check_consistency(enc_dir1, enc_dir2):
        mp1 = _get_model_name(enc_dir1)
        mp2 = _get_model_name(enc_dir2)
        if mp1 != mp2:
            raise RuntimeError("Inconsistent model names, %s and %s"%(mp1, mp2))
        et1 = _get_enc_transform(enc_dir1)
        et2 = _get_enc_transform(enc_dir2)
        if et1 != et2:
            raise RuntimeError("Inconsistent encoding transforms, %s and %s" \
                    % (et1, et2))

    def _check_paths(enc_dir):
        paths = [pj(enc_dir, "model_path.txt"), pj(enc_dir, 
            "encoding_chip_list.txt"), pj(enc_dir, "encoding_transform.txt"),
            pj(enc_dir, "encoding.npy")]
        for p in paths:
            if not pe(p):
                raise RuntimeError("Expecting to find path %s but it does not "\
                        "exist" % (p))

    def _get_enc_transform(enc_dir):
        with open( pj(enc_dir, "encoding_transform.txt") ) as fp:
            enc_tr = next(fp).strip()
        return enc_tr

    def _get_files(enc_dir):
        files = []
        with open( pj(enc_dir, "encoding_chip_list.txt") ) as fp:
            for line in fp:
                files.append( line.strip() )
        return files

    def _get_model_name(enc_dir):
        with open( pj(enc_dir, "model_path.txt") ) as fp:
            mp = next(fp).strip()
            while mp.endswith("/"):
                mp = mp[:-1]
            name = os.path.basename(mp)
        return name

    def _write_ancillary(enc_dir1, enc_dir2, output_dir):
        shutil.copy( pj(enc_dir1, "encoding_transform.txt"), output_dir )
        shutil.copy( pj(enc_dir1, "model_path.txt"), output_dir )
        with open( pj(output_dir, "original_encodings.txt"), "w" ) as fp:
            fp.write(enc_dir1 + "\n")
            fp.write(enc_dir2 + "\n")

    def _write_chip_list(enc_dir1, enc_dir2, output_dir):
        files1 = _get_files(enc_dir1)
        files2 = _get_files(enc_dir2)
        files = files1 + files2
        with open( pj(output_dir, "encoding_chip_list.txt"), "w" ) as fp:
            for f in files:
                fp.write(f + "\n")

    def _write_encs(enc_dir1, enc_dir2, output_dir):
        encs1 = np.load( pj(enc_dir1, "encoding.npy") )
        encs2 = np.load( pj(enc_dir2, "encoding.npy") )
        encs = np.concatenate((encs1, encs2))
        np.save(pj(output_dir, "encoding.npy"), encs)

    
    _check_paths(enc_dir1)
    _check_paths(enc_dir2)
    _check_consistency(enc_dir1, enc_dir2)
    if not pe(output_dir):
        os.make_dirs(output_dir)
    _write_encs(enc_dir1, enc_dir2, output_dir)
    _write_chip_list(enc_dir1, enc_dir2, output_dir)
    _write_ancillary(enc_dir1, enc_dir2, output_dir)

