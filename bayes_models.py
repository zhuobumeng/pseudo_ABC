import os
import torch
import generators

import numpy as np
import torch.nn as nn

from decorators import auto_init_pytorch, auto_init_args
# from scipy.stats import multivariate_normal as MN


class data_loader_edit:
    @auto_init_args
    def __init__(self, data, batch_size=100):
        self.data = data[np.random.permutation(len(data))]

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def prepare(self):
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i: i + self.batch_size]


class base(nn.Module):
    def __init__(self, input_dim, gen_init, theta_y, disc_init, experiment):
        super(base, self).__init__()
        assert experiment.config.ghsize == 0
        self.expe = experiment
        self.eps = self.expe.config.eps
        self.disc_init = disc_init
        self.std = 0

        # print(theta_y.flatten())

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Set Discriminator Neural Net
        if self.expe.config.dhsize:
            self.netD = nn.Sequential(
                nn.Linear(input_dim, self.expe.config.dhsize),
                nn.Tanh(),
                nn.Linear(self.expe.config.dhsize, self.expe.config.osize)
            )
        else:
            # self.netD = nn.Sequential(
            #     nn.Linear(input_dim, self.expe.config.osize, bias=False))
            self.netD = nn.Sequential(
                nn.Linear(input_dim, self.expe.config.osize))

        self.netD = self.netD.to(self.device)
        # update D state, set initial as the output of freq
        # self.netD.load_state_dict(disc_init)

        self.gen_init = gen_init  # Start initial center
        # self.gen_init = self.to_var(gen_init)
        self.theta_y_numpy = theta_y  # Freq best est
        self.theta_y = nn.Parameter(self.to_var(theta_y))
        self.theta_y.requires_grad = True

        # Set Generator Model
        self.netG = generators.bayes_trivial(input_dim, self.device)

        # Parameter of interest
        # self.valid_theta = []
        # self.importance_weights = []
        self.all_delta = []
        self.all_theta = []
        self.all_weights = []
        self.all_grad_norm = []
        # self.gradlist = []
        self.all_loss = []

    # def compute_weights(self, theta, gen_init=None):
    #     if gen_init is None:
    #         gen_init = self.gen_init_numpy

    #     # pi_theta = MN.pdf(
    #     #     theta.detach().cpu().numpy(),
    #     #     mean=np.zeros_like(self.gen_init_numpy.flatten()),
    #     #     cov=np.eye(self.expe.config.isize))
    #     # theta_theta = MN.pdf(
    #     #     theta.detach().cpu().numpy(),
    #     #     mean=self.gen_init_numpy.flatten(),
    #     #     cov=np.eye(self.expe.config.isize) * (self.expe.config.sv ** 2))
    #     # return pi_theta / theta_theta
    #     theta_numpy = theta.clone().detach().cpu().numpy()
    #     return np.exp(-0.5 * (theta_numpy ** 2).sum() +
    #                   ((theta_numpy - gen_init) ** 2).sum() /
    #                   (2 * (self.expe.config.std ** 2)))
    #     # return np.exp(-0.5 * (theta_numpy ** 2).sum())

    def compute_weights(self, delta, gen_mean=0, gen_std=1):
        scale_delta = (delta - gen_mean) / gen_std
        out_ = 0.5 * (scale_delta ** 2).sum()
        return np.exp(out_)

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
        # for p in self.netD.parameters():
        #     p.data.clamp_(-self.expe.config.wt, self.expe.config.wt)

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

    def theta_2_grad(self, theta):
        self.netD.load_state_dict(self.disc_init)
        z = self.to_var(np.random.normal(
            size=(self.expe.config.data_size, self.expe.config.zsize))
            .astype("float32"))
        fake_x = self.netG(z, self.to_var(theta))
        for ds_ in range(self.expe.config.ds):
            loss = self.trainD(fake_x)
        self.zero_grad()
        grad = self.get_gradient(fake_x)
        grad_norm = np.linalg.norm(grad)
        return grad_norm, loss.item()

    def get_theta(self):
        for it in range(self.expe.config.n_iteration):
            # self.netD.load_state_dict(self.disc_init)
            new_delta = (np.random.uniform(
                -7, 7, size=self.gen_init.shape).astype("float32"))
            # std = 2.
            # new_delta = std * np.random.normal(
            #     size=self.gen_init.shape).astype("float32")
            new_theta = self.gen_init - new_delta / np.sqrt(
                self.expe.config.data_size)
            for _ in range(self.expe.config.nt_pertheta):
                grad_norm, loss = self.theta_2_grad(new_theta)
                # weight = np.exp(0.5 * (new_delta**2).sum() / (std**2))
                weight = 1.
                self.all_theta.append(new_theta)
                self.all_delta.append(new_delta)
                self.all_weights.append(weight)
                self.all_grad_norm.append(grad_norm)
                self.all_loss.append(loss)
            if it % 100 == 0:
                self.expe.log.info("Iteration " + str(it) + " finished!")
                self.expe.log.info("New delta:" + str(new_delta))

    def get_theta2(self, verbose=False):
        for it in range(self.expe.config.n_iteration):
            self.netD.load_state_dict(self.disc_init)
            # new_delta = (np.random.uniform(
            #     -7, 7, size=self.gen_init.shape).astype("float32"))
            new_delta = 3. * np.random.normal(
                size=self.gen_init.shape).astype("float32")
            # generate new theta & fake X
            # new_theta = self.gen_init - self.expe.config.std * \
            # self.to_var(np.random.normal(
            #     size=self.gen_init.size()).astype("float32"))
            new_theta = self.gen_init - new_delta / np.sqrt(
                self.expe.config.data_size)

            for _ in range(self.expe.config.nt_pertheta):
                z = self.to_var(np.random.normal(
                    size=(self.expe.config.data_size, self.expe.config.zsize))
                    .astype("float32"))
                fake_x = data_loader_edit(
                    self.netG(z, self.to_var(new_theta)),
                    self.expe.config.bsize)
                fake_x_batch = fake_x.prepare()
                grad, loss = 0, 0
                for _, fake_data in enumerate(fake_x_batch):
                    for ds_ in range(self.expe.config.ds):
                        loss_ = self.trainD(fake_data, verbose=False)
                    # only save the last step for each data batch
                    loss = loss + loss_.item() / len(fake_x)

                    self.zero_grad()

                    grad_ = self.get_gradient(fake_data)
                    grad = grad + (grad_ / len(fake_x)).astype("float32")

                grad_norm = np.linalg.norm(grad)
                # weight = self.compute_weights(new_theta)
                weight = 1.
                self.all_theta.append(new_theta)
                self.all_delta.append(new_delta)
                self.all_weights.append(weight)
                self.all_grad_norm.append(grad_norm)
                self.all_loss.append(loss)
            if it % 1000 == 0:
                self.expe.log.info("Iteration " + str(it) + " finished!")


class BAYES_GAN(base):
    """
    sample based on given theta
    """
    @auto_init_pytorch
    def __init__(self, input_dim, gen_init, theta_y, disc_init, experiment):
        super(BAYES_GAN, self).__init__(
            input_dim, gen_init, theta_y, disc_init, experiment)

    def trainD(self, x, verbose=False, *args, **kwargs):

        # generate new theta & fake X
        score1 = -torch.sigmoid(self.netD(x)).mean()
        z = self.to_var(np.random.normal(
            size=(self.expe.config.ssize, self.expe.config.zsize))
            .astype("float32"))
        score2 = torch.sigmoid(
            self.netD(self.netG(z, self.theta_y).detach())).mean()
        loss = score1 + score2
        self.optimize(self.optD, loss)
        if verbose:
            self.expe.log.info(
                "optimizing D, loss: {:.4f}".format(loss.item()))
        return loss

    def get_gradient(self, x, *args, **kwargs):
        z = self.to_var(np.random.normal(
            size=(self.expe.config.ssize, self.expe.config.zsize))
            .astype("float32"))
        score = self.netD(self.netG(z, self.theta_y))
        loss = -torch.sigmoid(score).mean()

        loss.backward()
        grad = self.theta_y.grad.cpu().detach().clone().numpy()
        # grad_norm = np.linalg.norm(grad)

        # return grad, grad_norm
        return grad


class BAYES_GAN_penalty(base):
    @auto_init_pytorch
    def __init__(self, input_dim, gen_init, theta_y, disc_init, experiment):
        super(BAYES_GAN_penalty, self).__init__(
            input_dim, gen_init, theta_y, disc_init, experiment)

    def trainD(self, x, verbose=False, *args, **kwargs):

        # generate new theta & fake X
        score1 = -torch.sigmoid(self.netD(x)).mean()
        z = self.to_var(np.random.normal(
            size=(self.expe.config.ssize, self.expe.config.zsize))
            .astype("float32"))

        score2 = torch.sigmoid(
            self.netD(self.netG(z, self.theta_y).detach())).mean()

        loss = score1 + score2

        # penalty version
        if self.expe.config.pt_all:
            for netD_ in self.netD:
                if hasattr(netD_, "weight"):
                    loss = loss + self.expe.config.pt * (
                        netD_.weight ** 2).sum()
        else:
            loss = loss + self.expe.config.pt * (
                self.netD[-1].weight ** 2).sum()

        return loss

    def get_gradient(self, x, *args, **kwargs):
        z = self.to_var(np.random.normal(
            size=(self.expe.config.ssize, self.expe.config.zsize))
            .astype("float32"))
        score = self.netD(self.netG(z, self.theta_y))
        loss = -torch.sigmoid(score).mean()

        loss.backward()
        grad = self.theta_y.grad.cpu().detach().clone().numpy()
        # grad_norm = np.linalg.norm(grad)

        # return grad, grad_norm
        return grad


class BAYES_JS_GAN(base):
    @auto_init_pytorch
    def __init__(self, input_dim, gen_init, theta_y, disc_init, experiment):
        super(BAYES_JS_GAN, self).__init__(
            input_dim, gen_init, theta_y, disc_init, experiment)

    def trainD(self, x, verbose=False, *args, **kwargs):

        # generate new theta & fake X
        score1 = torch.log(torch.sigmoid(self.netD(self.to_var(x)))).mean()
        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score2 = torch.log(1 - torch.sigmoid(
            self.netD(self.netG(z, self.theta_y).detach()))).mean()
        loss = - score1 - score2 - np.log(4)
        self.optimize(self.optD, loss)

        if verbose:
            self.expe.log.info(
                "optimizing D, loss: {:.4f}".format(loss.item()))
        return loss

    def get_gradient(self, x, *args, **kwargs):
        z = self.to_var(np.random.normal(
            size=(self.expe.config.ssize, self.expe.config.zsize))
            .astype("float32"))
        score = torch.log(1 - torch.sigmoid(
            self.netD(self.netG(z, self.theta_y)))).mean()
        loss = score

        loss.backward()
        grad = self.theta_y.grad.detach().clone().cpu().numpy()
        # grad_norm = np.linalg.norm(grad)

        # return grad, grad_norm
        return grad
