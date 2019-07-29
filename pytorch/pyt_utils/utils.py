"""
PyTorch-based utilities
"""

import numpy as np
import os
import sys

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, \
        resnet152
from tensorboardX import SummaryWriter

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


# https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
def find_lr(model, train_loader, optimizer, criterion, init_value=1e-8,
        final_value=10.0, beta=0.98):
    num = len(train_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_loader:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.data[0]
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


# Inputs
#   resnet: Name of ResNet version
#   num_classes: Number of classes in final fc output layer
# Output
#   Returns the model pretrained on ImageNet with optional new final fc layer,
#   on cpu.
def get_resnet_model(resnet, num_classes=-1):
    if resnet == "resnet18":
        model = resnet18(pretrained=True)
    elif resnet == "resnet34":
        model = resnet34(pretrained=True)
    elif resnet == "resnet50":
        model = resnet50(pretrained=True)
    elif resnet == "resnet101":
        model = resnet101(pretrained=True)
    elif resnet == "resnet152":
        model = resnet152(pretrained=True)
    else:
        raise RuntimeError("Unsupported model, %s" % resnet)
    if num_classes > 0:
        in_features = model.fc.in_features
        fc = nn.Linear(in_features, num_classes, bias=True)
        model.fc = fc
    return model

# Inputs
#   session_dir: Session directory
# Output
#   Returns tensorboardX SummaryWriter object
def get_summary_writer(session_dir):
    writer_path = pj( session_dir, "tensorboard" )
    if not pe(writer_path):
        os.makedirs(writer_path)
    return SummaryWriter(writer_path)

# Inputs
#   model: The actual model (nn.Module subclass)
#   model_name: Name of model
#   epoch: Epoch number
#   models_dir: Directory holding the model
#   max_model_ct: Maximum number of models the directory can hold before old
#       ones start getting popped
# Output
#   Returns the path to the just-saved model.  Also may mutate the contents of
#       models_dir, by deleting the oldest model there.
def save_model_pop_old(model, model_name, epoch, models_dir, max_model_ct=5):
    min_ct = np.inf
    min_name = ""
    ibeg = len(model_name) + 1
    for name in os.listdir(models_dir):
        stub = os.path.splitext(name)[0]
        num = int( stub[ibeg:] )
        if num<min_ct:
            min_ct = num
            min_name = name
    if len( [f for f in os.listdir(models_dir) \
            if f.endswith(".pkl")] ) >= max_model_ct:
        os.remove( pj(models_dir, min_name) )
    path = pj(models_dir, model_name + "_%04d.pkl" % (epoch+1))
    torch.save(model.state_dict(), path)
    return path

