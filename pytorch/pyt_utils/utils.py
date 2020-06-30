"""
PyTorch-based utilities
"""

import numpy as np
import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.densenet import densenet121, densenet169, densenet201,\
        densenet161
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101,\
        resnet152
from tensorboardX import SummaryWriter

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


# Summary writer for tensorboard
g_tboard_writer = None

# Inputs
#   tboard_supdir: Parent directory of tensorboard directory, usually session
#       directory (session_dir)
# Output
#   Summary writer object
def create_and_set_summary_writer(tboard_supdir):
    global g_tboard_writer
    g_tboard_writer = get_summary_writer(tboard_supdir)
    return g_tboard_writer

# https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
# Example initialization and visualization:
#
#    net = SimpleNeuralNet(28*28,100,10)
#    optimizer = optim.SGD(net.parameters(),lr=1e-1)
#    criterion = F.nll_loss
#    
#    logs,losses = find_lr()
#    plt.plot(logs[10:-5],losses[10:-5])
def find_lr(model, train_loader, optimizer, criterion,
        data_splitter=lambda data : data,
        init_value=1e-8, final_value=10.0, beta=0.98):
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
        inputs,labels = data_splitter(data)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses

# Inputs
#   densenet: Name of DenseNet version
#   num_classes: Number of classes in final fc output layer
# Output
#   Returns the model pretrained on ImageNet with optional new final fc layer,
#   on cpu.
def get_densenet_model(densenet, num_classes=-1):
    if densenet == "densenet121":
        model = densenet121(pretrained=True)
    elif densenet == "densenet169":
        model = densenet169(pretrained=True)
    elif densenet == "densenet201":
        model = densenet201(pretrained=True)
    elif densenet == "densenet161":
        model = densenet161(pretrained=True)
    else:
        raise RuntimeError("Unsupported model, %s" % densenet)
    if num_classes > 0:
        in_features = model.classifier.in_features
        classifier = nn.Linear(in_features, num_classes, bias=True)
        model.classifier = classifier 
    return model

# Get optimizer used in model training
# Input
#   cfg (dict): Configuration parameters
#   lr (float): Learning rate
#   model_or_params: PyTorch module (nn.Module) or list/generator of parmaters
# Output
#   optimizer: torch.optim object
def get_optimizer(cfg, lr, model_or_params):
    if hasattr(model_or_params, "parameters"):
        params = [p for p in model_or_params.parameters() \
                if p.requires_grad]
    else:
        params = model_or_params
    if cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam( params, betas=(cfg["b1"], cfg["b2"]),
                eps=cfg["eps"], weight_decay=cfg["weight_decay"], lr=lr)
    elif cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(params, lr, momentum=cfg["momentum"],
                weight_decay=cfg["weight_decay"])
    else:
        raise NotImplementedError( cfg["optimizer"] )
    return optimizer

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
#   tboard_supdir: Parent directory of tensorboard directory, usually session
#       directory (session_dir)
# Output
#   Returns tensorboardX SummaryWriter object
def get_summary_writer(tboard_supdir):
    writer_path = pj( tboard_supdir, "tensorboard" )
    if not pe(writer_path):
        os.makedirs(writer_path)
    return SummaryWriter(writer_path)

# Inputs
# Output
#   Returns global tensorboardX SummaryWriter object
def get_tboard_writer():
    return g_tboard_writer

# Prints out simple basic statistics (min, max, mean, median) for a given
# variable.
# Inputs:
#   x: variable
#   name (optional): variable name
# Output:
#   None
def print_var_stats(x, name=None):
    s = "" if name is None else "%s, " % name
    B = torch if type(x)==torch.Tensor else np
    print("%sshape: %s, min: %f, max: %f, mean: %f, median: %f" %
            (s, repr(tuple(x.shape)), B.min(x), B.max(x), B.mean(x),
                B.median(x)))

# Load a state dict into a model
# Inputs
#   model: PyTorch model
#   model_path: Path to state dict
#   cudev: cuda device number or "cpu"
#   mode (optional): train or test
# Output
#   PyTorch model
def load_state_dict(model, model_path, cudev, mode="test"):
    sd = torch.load(model_path, map_location=lambda storage,loc : storage)
    model.load_state_dict(sd)
    if type(cudev)==int and cudev < 0:
        model = model.to("cpu")
    else:
        model = model.to(cudev)
    if mode=="test":
        model.eval()
        model.train(False)
    else:
        model.train(True)
    return model

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

# Inputs
#   model: PyTorch model, nn.Module subclass
#   writer: tensorboardX writer object
#   iter_ct: What model iteration
#   splitter: What character splits the layer names into components
# Output:
#   None
def write_tboard_layers(model, writer, iter_ct, splitter="."):
    # Expecting names like _encoder.3.weight, _decoder.9.bias, etc.
    grad_metrics = OrderedDict()
    for name,param in model.named_parameters():
        if model.get_is_conv_layer(name) and "weight" in name:
            HT,WD,wd,ht = min(16,param.shape[0]), min(16,param.shape[1]), \
                    param.shape[2], param.shape[3]
            img = torch.FloatTensor(1, HT*ht, WD*wd).to( param.device )
            for i in range(HT):
                for j in range(WD):
                    img[0, i*ht:(i+1)*ht, j*wd:(j+1)*wd] = param[i,j]
            writer.add_image(name, img, iter_ct)

        name_tokens = name.split(splitter)
        s = ""
        if name_tokens[0] == "_encoder":
            cat = "Encoder"
            name = splitter.join( name_tokens[1:] )
        elif name_tokens[0] == "_decoder":
            cat = "Decoder"
            name = splitter.join( name_tokens[1:] )
        elif "bneck" in name_tokens:
            cat = "Bottleneck"
            name = splitter.join(name_tokens)
        else:
            continue
        s += cat + "/"

        pows10 = [2,4,6]
        if cat not in grad_metrics:
            grad_metrics[cat] = OrderedDict()
            grad_metrics[cat]["SD"] = OrderedDict()
            for e in pows10:
                grad_metrics[cat]["Pct%d" % e] = OrderedDict()
        pgrad = param.grad
        if "weight" in name_tokens:
            grad_metrics[cat]["SD"][name] = torch.std(pgrad)
            for e in pows10:
                grad_metrics[cat]["Pct%d" % e][name] = 100.0 * torch.mean( \
                        (torch.abs(pgrad) < pow(10,-e)).float() )

        if "weight" in name_tokens:
            s += "weight/"
        elif "bias" in name_tokens:
            s += "bias/"

        writer.add_histogram(s+name, param, iter_ct)
        writer.add_histogram("Gradient/"+s+name, param.grad, iter_ct)

    for cat,cat_dict in grad_metrics.items():
        for met_name,met_dict in cat_dict.items():
            writer.add_scalars("Gradient/"+cat+"/"+met_name, met_dict,
                    iter_ct)

