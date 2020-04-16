import inspect
import numpy as np
from copy import deepcopy


def auto_init_args(init):
    def new_init(self, *args, **kwargs):
        arg_dict = inspect.signature(init).parameters
        arg_names = list(arg_dict.keys())[1:]  # skip self
        proc_names = set()
        for name, arg in zip(arg_names, args):
            setattr(self, name, arg)
            proc_names.add(name)
        for name, arg in kwargs.items():
            setattr(self, name, arg)
            proc_names.add(name)
        remain_names = set(arg_names) - proc_names
        if len(remain_names):
            for name in remain_names:
                setattr(self, name, arg_dict[name].default)
        init(self, *args, **kwargs)

    return new_init


def auto_init_pytorch(init):
    def new_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.optD = self.init_optimizer(
            self.expe.config.opt,
            self.expe.config.lr_d,
            self.netD.parameters())

        if list(self.netG.parameters()):
            self.optG = self.init_optimizer(
                self.expe.config.opt,
                self.expe.config.lr_g,
                self.netG.parameters())

        if not self.expe.config.resume:
            self.to(device=self.device)
            self.expe.log.info(
                "transferred model to {}".format(self.device))

    return new_init


class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = 10000
        self.early_stop = False

    def __call__(self, loss, model):
        if loss.item() >= self.best_loss:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
                # if self.verbose:
                #     model.expe.log.info("Early stop: out of patience 10.")
        else:
            self.netD_checkpoint = deepcopy(model.state_dict())
            self.best_loss = loss.item()
            self.counter = 0
