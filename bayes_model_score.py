import os
import torch
# import generators

import numpy as np
import torch.nn as nn

# from decorators import auto_init_pytorch
# from scipy.stats import multivariate_normal as MN


class base(nn.Module):
    def __init__(self, input_dim, experiment):
        super(base, self).__init__()
        self.expe = experiment
        self.eps = self.expe.config.eps
        self.dim = input_dim

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Parameter of interest
        self.all_delta = []
        self.all_theta = []
        self.all_weights = []
        self.all_grad_norm = []
        # self.all_grad_maxnorm = []
        # self.gradlist = []
        self.all_loss = []

    def compute_weights(self, theta, delta, gen_mean=0, gen_std=1):
        scale_delta = (delta - gen_mean) / gen_std
        # n = -0.5 * (theta ** 2).sum() + 0.5 * (scale_delta ** 2).sum()
        n = 0.5 * (scale_delta ** 2).sum()
        # print("n before", n)
        # n = n - self.expe.config.isize / 2 * np.log(self.expe.config.data_size)
        # n = n + 1/2 * (np.log(gen_std)).sum()
        return np.exp(n)

    # def compute_weights(self, theta, gen_init=None):
    #     if gen_init is None:
    #         gen_init = self.gen_init_numpy
    #     theta_numpy = theta.clone().detach().cpu().numpy()
    #     return np.exp(-0.5 * (theta_numpy ** 2).sum() +
    #                   ((theta_numpy - gen_init) ** 2).sum() /
    #                   (2 * (self.expe.config.std ** 2)))

    def to_var(self, inputs):
        if torch.is_tensor(inputs):
            return inputs.clone().detach().to(device=self.device)
        else:
            return torch.tensor(inputs, device=self.device)

    def to_vars(self, *inputs):
        return [self.to_var(inputs_) if inputs_ is not None and inputs_.size
                else None for inputs_ in inputs]

    def optimize(self, opt, loss):
        self.zero_grad()
        loss.backward()
        opt.step()

    def init_optimizer(self, opt_type, learning_rate, param):
        if opt_type.lower() == "adam":
            optimizer = torch.optim.Adam
        elif opt_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop
        elif opt_type.lower() == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError("invalid optimizer: {}".format(opt_type))

        opt = optimizer(
            params=filter(
                lambda p: p.requires_grad, param
            ),
            lr=learning_rate)

        return opt

    def save(self, name="latest"):
        save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
        checkpoint = {
            "state_dict": self.state_dict(),
            # "theta": self.valid_theta,
            "config": self.expe.config
        }
        torch.save(checkpoint, save_path)
        self.expe.log.info("model saved to {}".format(save_path))

    def load(self, checkpointed_state_dict=None, name="latest"):
        if checkpointed_state_dict is None:
            save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
            checkpoint = torch.load(save_path,
                                    map_location=lambda storage,
                                    loc: storage)
            self.load_state_dict(checkpoint['state_dict'])
            self.expe.log.info("model loaded from {}".format(save_path))
        else:
            self.load_state_dict(checkpointed_state_dict)
            self.expe.log.info("model loaded!")

    def get_theta(self, theta_y):

        self.theta_y = theta_y
        dim = self.expe.config.isize

        for it in range(self.expe.config.n_iteration):
            # new_delta = np.random.normal(
            #     size=(1, dim)).astype("float32")
            new_delta = np.random.uniform(
                -10, 10, size=(1, dim)).astype("float32")
            # self.expe.log.info("new_delta:" + str(new_delta))
            new_theta = self.theta_y + new_delta / np.sqrt(
                self.expe.config.data_size)
            # self.expe.log.info("new_theta:" + str(new_theta))
            for _ in range(self.expe.config.nt_pertheta):
                fake_x = self.gen_data(new_theta, self.expe.config.data_size)
                grad = self.score_fun(fake_x, self.theta_y)
                grad_norm = np.linalg.norm(grad)
                # self.expe.log.info("gradnorm:" + str(grad_norm))
                # grad_maxnorm = np.absolute(grad).max()

                # weight = self.compute_weights(
                #     new_theta, new_delta, gen_std=self.expe.config.std)
                weight = 1.
                self.all_theta.append(new_theta)
                self.all_delta.append(new_delta)
                self.all_weights.append(weight)
                self.all_grad_norm.append(grad_norm)
                # self.all_grad_maxnorm.append(grad_maxnorm)


class exp_dist(base):
    def __init__(self, input_dim, experiment):
        super(exp_dist, self).__init__(
            input_dim, experiment)

    def gen_data(self, theta, data_size):
        # size: n x dim
        fake_x = np.random.exponential(
            scale=theta,
            size=(data_size, self.dim)).astype("float32")
        return fake_x

    def score_fun(self, data, theta):
        """
        data: n x dim
        theta: 1 x dim
        """
        assert (data.shape[1] == self.dim) and (
            theta.shape == self.theta_y.shape), "Not match dimension"
        score = (data.mean(0)[None, :] - theta) / (theta ** 2)
        return score

    def estimate(self, data):
        return data.mean(0)[None, :]


class gaussian_dist(base):
    def __init__(self, input_dim, experiment):
        super(gaussian_dist, self).__init__(
            input_dim, experiment)

    def gen_data(self, theta, data_size):
        z = np.random.normal(size=(data_size, self.dim)).astype("float32")
        fake_x = z + theta
        return fake_x

    def score_fun(self, data, theta):
        """
        data: n x dim
        theta: 1 x dim
        """
        assert (data.shape[1] == self.dim) and (
            theta.shape == self.theta_y.shape), "Not match dimension"
        score = data.mean(0)[None, :] - theta
        return score

    def estimate(self, data):
        return data.mean(0)[None, :]


class bernoulli_dist(base):
    def __init__(self, input_dim, experiment):
        super(bernoulli_dist, self).__init__(
            input_dim, experiment)

    def gen_data(self, theta, data_size):
        # size: n x dim
        fake_x = np.random.binomial(
            n=1, p=theta,
            size=(data_size, self.dim)).astype("float32")
        return fake_x

    def score_fun(self, data, theta):
        """
        data: n x dim
        theta: 1 x dim
        """
        assert (data.shape[1] == self.dim) and (
            theta.shape == self.theta_y.shape), "Not match dimension"
        score = (data.mean(0)[None, :] - theta) / (theta * (1 - theta))
        return score

    def estimate(self, data):
        return data.mean(0)[None, :]


class poisson_dist(base):
    def __init__(self, input_dim, experiment):
        super(poisson_dist, self).__init__(
            input_dim, experiment)

    def gen_data(self, theta, data_size):
        # size: n x dim
        fake_x = np.random.poisson(
            lam=theta,
            size=(data_size, self.dim)).astype("float32")
        return fake_x

    def score_fun(self, data, theta):
        """
        data: n x dim
        theta: 1 x dim
        """
        assert (data.shape[1] == self.dim) and (
            theta.shape == self.theta_y.shape), "Not match dimension"
        score = (data.mean(0)[None, :] - theta) / theta
        return score

    def estimate(self, data):
        return data.mean(0)[None, :]
