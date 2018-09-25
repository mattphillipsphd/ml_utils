"""
A generic trainer class.
"""

import numpy as np
import os
import pathlib
import shutil
import torch
import torch.nn as nn

from collections import OrderedDict
from torch.autograd import Variable
from torchvision.utils import save_image

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


# This sampler is appropriate for image autoencoder models
def ae_sampler(epoch, trainer):
    trainer._model.eval()
    img_size = trainer._model.get_input_size()
    sample_batch_size = 32 # TODO
    inputs = []
    targets = []
    for i in range(sample_batch_size):
        inp,target = trainer._train_loader.dataset.__getitem__(i) # TODO test
        inputs.append(inp)
        targets.append(target)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    outputs = trainer._model( Variable(inputs).cuda() )[0]
    output_imgs = [o.data.cpu() for o in torch.squeeze(outputs)]
    output_imgs = torch.stack(output_imgs).view(sample_batch_size, 1, img_size,
            img_size)

    comparisons = torch.cat((Variable(inputs), Variable(output_imgs),
        Variable(targets)))
    save_image(comparisons, pj(trainer._samples_dir, "comparisons_%03d.png" \
            % (epoch)))


def get_session_dir(output_supdir):
    stub = pj(output_supdir, "session_%02d")
    ct = 0
    while pe(stub % (ct)):
        if pe(pj(stub % (ct), "delete_me.txt")):
            shutil.rmtree(stub % (ct))
            break
        ct += 1
    os.makedirs(stub % (ct))
    pathlib.Path( pj(stub % (ct), "delete_me.txt") ).touch()
    return stub % (ct)


class Trainer():
    def __init__(self, model, loaders, opt_getter, criterion, session_dir,
            num_epochs=100, base_lr=0.001, num_lr_drops=2, lr_drop_factor=5, 
            log_interval=10, epoch_writer=None):
        self._base_lr = base_lr
        self._criterion = criterion
        self._lr_drop_factor = lr_drop_factor
        self._epoch_writer = epoch_writer
        self._is_initialized = False
        self._last_avg_loss = None
        self._loaders = loaders
        self._log_interval = log_interval
        self._lr = None
        self._lr_drop_ct = None
        self._model = model
        self._model_dir = None
        self._num_epochs = num_epochs
        self._num_lr_drops = num_lr_drops
        self._optimizer = None
        self._opt_getter = opt_getter
        self._session_dir = session_dir
        self._test_loader = loaders[1]
        self._test_loss = None
        self._train_loader = loaders[0]
        self._train_loss = None
        

    def train(self):
        self._init_session()
        for epoch in range(self._num_epochs):
            self._model.train()
            self._train_loss = 0
            for batch_idx, (inputs, targets) in enumerate(self._train_loader):
                x = Variable(inputs).cuda() # TODO, iterate over tuple
                y = Variable(targets).cuda()
                self._optimizer.zero_grad()
                yhat = self._model(x)
                loss = self._criterion(yhat, y)
                loss.backward()
                self._train_loss += loss.item()
                self._optimizer.step()
                # TODO add in train_step, test_step interface

                if batch_idx % self._log_interval == 0: # TODO add in test loss
                    s = "\t\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n"\
                            .format(epoch,
                                batch_idx * len(x),
                                len(self._train_loader.dataset),
                                100. * batch_idx / len(self._train_loader),
                                loss.item() / len(x))
                    with open(pj(self._session_dir, "session.log"), "a") as fp:
                        fp.write(s)
                        fp.flush

            avg_loss = self._train_loss / len(self._train_loader.dataset)
            s = "====> Epoch: {} Average loss: {:.4f}".format(epoch, avg_loss)
            with open(pj(self._session_dir, "session.log"), "a") as fp:
                fp.write(s + "\n")
                fp.flush()
            print(s)

            self._write_epoch(epoch)
            if avg_loss > self._last_avg_loss:
                if self._lr_drop_ct == self._num_lr_drops:
                    break
                self._lr = self._lr / self._lr_drop_factor
                self._optimizer = self._get_optimizer()
                print("Learning rate dropped to %f" % (self._lr))
                self._lr_drop_ct += 1
            else:
                self._last_avg_loss = avg_loss
                self._save_model()

            self._retain_session()

    def _init_session(self):
        if self._is_initialized:
            raise RuntimeError("_init_session called, but training session " \
                    "already initialized")
        self._lr = self._base_lr
        self._lr_drop_ct = 0
        self._last_avg_loss = np.Inf
        self._optimizer = self._get_optimizer()
        self._train_loss = 0
        self._test_loss = 0

        self._model_dir = pj(self._session_dir, "models")
        os.makedirs(self._model_dir)
        self._samples_dir = pj(self._session_dir, "samples")
        os.makedirs(self._samples_dir)
        self._is_initialized = True

    def _get_optimizer(self):
        return self._opt_getter(self._lr)

    def _retain_session(self):
        if pe(pj(self._session_dir, "delete_me.txt")):
            os.remove( pj(self._session_dir, "delete_me.txt") )

    def _save_model(self):
        path = pj(self._model_dir, "model.pkl")
        torch.save(self._model.state_dict(), path)

    def _write_epoch(self, epoch):
        if self._epoch_writer is None:
            return
        if not callable(self._epoch_writer):
            raise RuntimeError("Parameter epoch_writer must be either None or "\
                    "a callable function")
        self._epoch_writer(epoch, self)

