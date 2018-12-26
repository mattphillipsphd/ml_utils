"""
A generic trainer class.
"""

import abc
import numpy as np
import os
import pathlib
import shutil
import torch
import torch.nn as nn

from collections import OrderedDict

# pytorch imports
from torch.autograd import Variable
from torchvision.utils import save_image

# local imports

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

g_delete_me_txt = "delete_me.txt"
g_session_dir_stub = "session_%02d"
g_session_log = "session.log"


def class_sampler(epoch, trainer):
    model = trainer.get_model()
    model.eval()
    dataset = trainer.get_train_loader().dataset # TODO should be a test image
    data_loader = trainer.get_train_loader()
    output_img = np.zeros((500,500))
    cats_dict = dataset.get_cats_dict()
    inv_cats_dict = OrderedDict()
    for k,v in cats_dict.items():
        inv_cats_dict[v] = k
    inv_cats_dict[0] = 0
    output_img = []
    for batch_idx,(data,_) in enumerate(data_loader):
        data = Variable(data).cuda()
        outputs = model(data)
        for d in np.array( outputs.cpu().data ):
            output_img.append( inv_cats_dict[ np.argmax(d) ] )
    output_img = np.resize(np.array(output_img), (500,500))
    output_img = torch.Tensor(output_img) / 256.0

    save_image(output_img, pj(trainer._samples_dir, "classifications_%03d.png" \
            % (epoch)))


class TrainerBase():
    def __init__(self, model, loaders, session_dir, num_epochs=1000,
            base_lr=0.001, num_lr_drops=2, lr_drop_factor=5, log_interval=10,
            use_cuda=True):
        self._base_lr = base_lr
        self._lr_drop_factor = lr_drop_factor
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
        self._session_dir = session_dir
        self._test_acc_loss = None
        self._test_loader = loaders[1]
        self._test_loader_iter = None
        self._test_loss = None
        self._train_acc_loss = None
        self._train_loader = loaders[0]
        self._train_loader_iter = None
        self._test_loss = None
        self._use_cuda = use_cuda

        if type(self._test_loader) is not torch.utils.data.DataLoader:
            raise RuntimeError("A test dataloader must be supplied")

    def get_model(self):
        return self._model

    def get_session_dir(self):
        return self._session_dir

    def get_test_loader(self):
        return self._test_loader

    def get_train_loader(self):
        return self._train_loader

    def train(self):
        self._init_session()
        for epoch in range(self._num_epochs):
            self._train_acc_loss = 0
            self._test_acc_loss = 0
            test_batch_idx = 0
            for batch_idx, (inputs, targets) in enumerate(self._train_loader):
                self._model.train()
                if self._use_cuda:
                    x = Variable(inputs).cuda() # TODO, iterate over tuple
                    y = Variable(targets).cuda()
                else:
                    x = Variable(inputs) # TODO, iterate over tuple
                    y = Variable(targets)
                self._optimizer.zero_grad()
                yhat = self._model(x)
                losses = self._criterion(yhat, y)
                loss = losses[0]
                self._train_loss = loss.item()
                self._train_acc_loss += loss.item()
                loss.backward()
                self._optimizer.step()
                # TODO add in train_step, test_step interface

                if batch_idx % self._log_interval == 0: # TODO add in test loss
                    self._model.eval()
                    test_inputs,test_targets \
                            = self._get_test_data(test_batch_idx)
                    if self._use_cuda:
                        test_x = Variable(test_inputs).cuda()
                        test_y = Variable(test_targets).cuda()
                    else:
                        test_x = Variable(test_inputs)
                        test_y = Variable(test_targets)
                    self._optimizer.zero_grad()
                    test_yhat = self._model(test_x)
                    test_losses = self._criterion(test_yhat, test_y)
                    test_loss = test_losses[0]
                    self._test_loss = test_loss.item()
                    self._test_acc_loss += test_loss.item()
                    self._write_batch(epoch, batch_idx, test_batch_idx)
                    test_batch_idx += 1

                self._post_batch_hook()

            self._write_epoch(epoch, test_batch_idx)
            avg_loss = self._test_acc_loss / test_batch_idx
            if self._last_avg_loss is None:
                self._last_avg_loss = avg_loss
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

    # This function is primarily for testing purposes
    def train_one(self):
        if not self._is_initialized:
            self._init_session()
            self._train_loader_iter = iter(self._train_loader)
        batch_idx = 0
        self._model.train()
        inputs,targets = next(self._train_loader_iter)
        if self._use_cuda:
            x = Variable(inputs).cuda() # TODO, iterate over tuple
            y = Variable(targets).cuda()
        else:
            x = Variable(inputs)
            y = Variable(targets)
        self._optimizer.zero_grad()
        yhat = self._model(x)
        losses = self._criterion(yhat, y)
        loss = losses[0]
        self._train_loss = loss.item()
        self._train_acc_loss += loss.item()
        loss.backward()
        self._optimizer.step()
        return x,y,yhat,loss

    @abc.abstractmethod
    def _criterion(self, yhat, y):
        return None

    def _init_session(self):
        if self._is_initialized:
            raise RuntimeError("_init_session called, but training session " \
                    "already initialized")
        self._lr = self._base_lr
        self._lr_drop_ct = 0
        self._optimizer = self._get_optimizer()
        self._test_acc_loss = 0
        self._test_loader_iter = iter(self._test_loader)
        self._train_acc_loss = 0

        self._model_dir = pj(self._session_dir, "models")
        os.makedirs(self._model_dir)
        self._samples_dir = pj(self._session_dir, "samples")
        os.makedirs(self._samples_dir)
        self._is_initialized = True

    @abc.abstractmethod
    def _get_optimizer(self):
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self._lr, 
                momentum=0.9)
        return optimizer

    def _get_test_data(self, test_batch_idx):
        batch_len = self._test_loader.batch_size
        if test_batch_idx * batch_len > len(self._test_loader.dataset):
            self._test_loader_iter = iter(self._test_loader)
        data,labels = next(self._test_loader_iter)
        return data,labels

    # Intended to be overridden in derived class
    def _post_batch_hook(self):
        pass

    def _save_model(self):
        path = pj(self._model_dir, "%s.pkl" % (self._model.get_name()))
        torch.save(self._model.state_dict(), path)

    @abc.abstractmethod
    def _write_batch(self, epoch, batch_idx, test_batch_idx):
        def _write_line(mode, epoch, batch_idx, batch_len, loss, loader):
            s = "\t\t{} Epoch: {} [{}/{} ({:.0f}%)]\t{} Loss: {:.6f}\n" \
                    .format(mode, epoch,
                        batch_idx * batch_len,
                        len(loader.dataset),
                        100. * batch_idx / len(loader),
                        mode,
                        loss / batch_len)
            with open(pj(self.get_session_dir(), g_session_log), "a") as fp:
                fp.write(s)
                fp.flush()

        batch_len = self.get_train_loader().batch_size # Can be wrong on last
            # batch
        _write_line("Train", epoch, batch_idx, batch_len, self._train_loss,
                self.get_train_loader())
        _write_line("Test", epoch, test_batch_idx, batch_len, self._test_loss,
                self.get_test_loader())

    @abc.abstractmethod
    def _write_epoch(self, epoch, test_batch_ct):
        avg_train_loss = self._train_acc_loss / len(self._train_loader.dataset)
        avg_test_loss = self._test_acc_loss / test_batch_ct
        s = "====> Epoch: {} Avg train loss: {:.4f} || test loss: {:.4f}"\
                .format(epoch, avg_train_loss, avg_test_loss)
        with open(pj(self._session_dir, g_session_log), "a") as fp:
            fp.write(s + "\n")
            fp.flush()
        print(s)

