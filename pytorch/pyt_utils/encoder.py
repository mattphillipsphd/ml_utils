"""
Generates features (encodings) from a model and a dataset
"""

import logging
import numpy as np
import os
import torch

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
    raise
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

    
