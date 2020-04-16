import os
import time
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plot

from decorators import auto_init_args
from config import get_base_parser


class tracker:
    @auto_init_args
    def __init__(self, names):
        assert len(names) > 0
        self.reset()

    def __getitem__(self, name):
        return self.values.get(name, 0) / self.counter[name] \
            if self.counter[name] else 0

    def __len__(self):
        return len(self.names)

    def reset(self):
        self.values = dict({name: 0. for name in self.names})
        self.counter = dict({name: 0. for name in self.names})
        self.create_time = time.time()

    def update(self, named_values, count):
        """
        named_values: dictionary with each item as name: value
        """
        for name, value in named_values.items():
            self.values[name] += value.item() * count
            self.counter[name] += count

    def summarize(self, output=""):
        if output:
            output += ", "
        for name in self.names:
            output += "{}: {:.3f}, ".format(
                name, self.values[name] / self.counter[name]
                if self.counter[name] else 0)
        output += "elapsed time: {:.1f}(s)".format(
            time.time() - self.create_time)
        return output

    @property
    def stats(self):
        return {n: v / self.counter[n] if self.counter[n] else 0
                for n, v in self.values.items()}


class experiment:
    @auto_init_args
    def __init__(self, config, experiments_prefix, logfile_name="log"):
        """Create a new Experiment instance.

        Modified based on: https://github.com/ex4sperans/mag

        Args:
            logfile_name: str, naming for log file. This can be useful to
                separate logs for different runs on the same experiment
            experiments_prefix: str, a prefix to the path where
                experiment will be saved
        """

        # get all defaults
        all_defaults = {}
        for key in vars(config):
            all_defaults[key] = get_base_parser().get_default(key)

        self.default_config = all_defaults

        # activation function
        activate_list = ["Tanh", "ReLU", "Softplus", "LogSigmoid"]
        map_activate = {act.lower(): act for act in activate_list}
        if self.config.activate.lower() in map_activate:
            self.config.activate = map_activate[self.config.activate.lower()]
        else:
            raise ValueError("Choose activation function among 'Tanh,'"
                             "ReLU, Softplus, logsigmoid'")

        config.resume = False
        if not config.debug:
            if os.path.isdir(self.experiment_dir):
                print("log exists: {}".format(self.experiment_dir))
                config.resume = True

            print(config)
            self._makedir()

    def _makedir(self):
        os.makedirs(self.experiment_dir, exist_ok=True)

    @property
    def experiment_dir(self):
        # if self.config.debug:
        #     return self.experiments_prefix + "/"
        # else:

        # get namespace for each group of args
        arg_g = dict()
        for group in get_base_parser()._action_groups:
            group_d = {a.dest: self.default_config.get(a.dest, None)
                       for a in group._group_actions}
            arg_g[group.title] = argparse.Namespace(**group_d)

        # skip default value
        identifier = ""
        for key, value in sorted(vars(arg_g["model_configs"]).items()):
            identifier += key + str(getattr(self.config, key))
        return os.path.join(self.experiments_prefix, identifier)

    @property
    def log_file(self):
        return os.path.join(self.experiment_dir, self.logfile_name)

    def register_directory(self, dirname):
        directory = os.path.join(self.experiment_dir, dirname)
        os.makedirs(directory, exist_ok=True)
        setattr(self, dirname, directory)

    def _register_existing_directories(self):
        for item in os.listdir(self.experiment_dir):
            fullpath = os.path.join(self.experiment_dir, item)
            if os.path.isdir(fullpath):
                setattr(self, item, fullpath)

    def __enter__(self):

        if self.config.debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M')
        else:
            print("log saving to", self.log_file)
            logging.basicConfig(
                filename=self.log_file,
                filemode='a+', level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M')

        self.log = logging.getLogger()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        logging.shutdown()

    @property
    def elapsed_time(self):
        return (time.time() - self.start_time) / 3600


def plot_theta(thetas, true_x, true_y, path):
    """
    thetas: list of 2d numy array
    """
    t = np.concatenate(thetas)
    plot.scatter(x=t[:, 0], y=t[:, 1], c='b')
    plot.scatter(x=true_x[0], y=true_y[0], c='r')
    plot.axvline(x=true_x[0], linestyle="--", c='r')
    plot.axhline(y=true_y[0], linestyle="--", c='r')
    plot.scatter(x=true_x[1], y=true_y[1], c='y')
    plot.axvline(x=true_x[1], linestyle="--", c='y')
    plot.axhline(y=true_y[1], linestyle="--", c='y')
    plot.savefig(path, format='png')
    plot.cla()


def plot_theta_dim1(thetas, true_x, path):
    """
    thetas: list of 1d numy array
    """
    t = np.concatenate(thetas)
    plot.hist(x=t, bins=50)
    plot.axvline(x=true_x[0], linestyle="--", c='r')
    plot.axvline(x=true_x[1], linestyle="--", c='y')
    plot.savefig(path, format='png')
    plot.cla()


def plot_theta_trajectory(thetas, true_x, true_y, path):
    """
    thetas: list of 2d numy array
    """
    t = np.concatenate(thetas)
    plot.plot(t[:, 0], t[:, 1], '-o', c='b')
    plot.scatter(x=true_x[0], y=true_y[0], c='r')
    plot.axvline(x=true_x[0], linestyle="--", c='r')
    plot.axhline(y=true_y[0], linestyle="--", c='r')
    plot.scatter(x=true_x[1], y=true_y[1], c='y')
    plot.axvline(x=true_x[1], linestyle="--", c='y')
    plot.axhline(y=true_y[1], linestyle="--", c='y')
    plot.savefig(path, format='png')
    plot.cla()


def plot_theta_2d_hist(thetas, true_x, true_y, weights, path):
    """
    thetas: list of 2d numy array
    """
    t = np.concatenate(thetas)
    plot.hist2d(x=t[:, 0], y=t[:, 1],
                weights=np.array(weights) / np.array(weights).sum(),
                bins=50, cmap=plot.cm.gray)
    plot.colorbar()
    plot.scatter(x=true_x[0], y=true_y[0], c='r')
    plot.axvline(x=true_x[0], linestyle="--", c='r')
    plot.axhline(y=true_y[0], linestyle="--", c='r')
    plot.scatter(x=true_x[1], y=true_y[1], c='y')
    plot.axvline(x=true_x[1], linestyle="--", c='y')
    plot.axhline(y=true_y[1], linestyle="--", c='y')
    plot.savefig(path, format='png')
    plot.clf()
