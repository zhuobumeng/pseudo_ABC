import os
import torch
import generators

import numpy as np
import torch.nn as nn

from decorators import auto_init_pytorch
# from scipy.stats import multivariate_normal as MN


class base(nn.Module):
    def __init__(self, gen_init, indices, disc_init, experiment):
        super(base, self).__init__()
        # gen_init: 1 x (dim x 4)
        self.expe = experiment
        self.gen_init = gen_init
        self.indices = indices
        self.disc_init = disc_init
        input_dim = self.expe.config.isize

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Set Discriminator Neural Net
        if self.expe.config.dhsize:
            self.netD = nn.Sequential(
                nn.Linear(input_dim, self.expe.config.dhsize),
                getattr(nn, self.expe.config.activate)(),
                nn.Linear(self.expe.config.dhsize, self.expe.config.osize)
            )
        else:
            self.netD = nn.Sequential(
                nn.Linear(input_dim, self.expe.config.osize))

        self.netD = self.netD.to(self.device)

        # update D state, set initial as the output of freq
        # self.netD.load_state_dict(disc_init)

        # Set Generator Model
        self.netG = generators.gnk(
            input_dim, gen_init, indices, self.device)

        # Parameter of interest
        self.all_delta = []
        self.all_theta = []
        self.all_weights = []
        self.all_grad_norm = []
        self.all_loss = []

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
            "theta": self.all_theta,
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

    def get_theta(self, verbose=False):
        for it in range(self.expe.config.n_iteration):
            # generate new theta & fake X
            # while True:
            #     new_delta = self.expe.config.std * np.random.normal(
            #         size=(1, self.dim)).astype("float32") * self.indc
            #     new_theta = self.gen_init - new_delta / np.sqrt(
            #         self.expe.config.data_size)
            #     if np.absolute(new_theta).max() < 10:
            #         break

            # # reset network before each sampling
            self.netD.load_state_dict(self.disc_init)
            _delta = (np.random.uniform(
                -5, 5, size=self.gen_init.shape).astype("float32"))

            new_delta = self.indices * _delta
            new_theta = self.gen_init - new_delta / np.sqrt(
                self.expe.config.data_size)
            # new_theta = np.array([3.01, 2, 1, 0.5])
            # print("new_theta", new_theta[0])
            # print("sum D weight", (self.netD[-1].weight ** 2).sum().item())

            for _ in range(self.expe.config.nt_pertheta):
                fake_x = self.to_var(self.gen_data(
                    new_theta, self.expe.config.data_size))
                for ds_ in range(self.expe.config.ds):
                    loss = self.trainD(fake_x, verbose=False)

                self.zero_grad()

                grad, grad_norm = self.get_gradient(fake_x)

                # weight = self.compute_weights(
                #     new_delta, gen_std=self.expe.config.std)
                weight = 1.
                self.all_theta.append(new_theta)
                self.all_delta.append(new_delta)
                self.all_weights.append(weight)
                self.all_grad_norm.append(grad_norm)
                self.all_loss.append(loss.item())
                # print("norm_grad", grad_norm)
                # print("sum D weight", (self.netD[-1].weight ** 2).sum().item())
                # print("loss", loss.item())
                # print("------")

            if it % 20 == 0:
                self.expe.log.info("iteration: {}".format(it))
                self.expe.log.info("new delta:" + str(new_delta.flatten()))
                self.expe.log.info("new theta:" + str(new_theta.flatten()))
                self.expe.log.info("grad:" + str(grad))
                self.expe.log.info("value of k is {:.3f}".format(self.netG.k.detach().clone().cpu().numpy()))
                self.expe.log.info("loss is {:.3f}".format(loss.item()))

    def gen_data(self, params, data_size=250):
        # not torch format
        # a, b, g, k: 1 * dim or scalar
        # output: n * dim
        a, b, g, k = params
        # print("gen data params", params)
        invec = np.random.normal(
            size=(data_size, self.expe.config.isize)).astype("float32")
        tmp = (1 - np.exp(- g * invec)) / (1 + np.exp(- g * invec))
        return a + b * (1 + 0.8 * tmp) * ((1 + invec**2)**k) * invec

# class BAYES_GAN(base):
#     """
#     sample based on given theta
#     """
#     @auto_init_pytorch
#     def __init__(self, input_dim, gen_init, disc_init, experiment):
#         super(BAYES_GAN, self).__init__(
#             input_dim, gen_init, disc_init, experiment)
#
#     def trainD(self, x, verbose=False, *args, **kwargs):
#
#         # generate new theta & fake X
#         score1 = -torch.sigmoid(self.netD(x)).mean()
#         z = self.to_var(np.random.normal(
#             size=(self.expe.config.ssize, self.expe.config.zsize))
#             .astype("float32"))
#         score2 = torch.sigmoid(
#             self.netD(self.netG(z, self.theta_y).detach())).mean()
#         loss = score1 + score2
#         self.optimize(self.optD, loss)
#         if verbose:
#             self.expe.log.info(
#                 "optimizing D, loss: {:.4f}".format(loss.item()))
#         return loss
#
#     def get_gradient(self, x, *args, **kwargs):
#         z = self.to_var(np.random.normal(
#             size=(self.expe.config.ssize, self.expe.config.zsize))
#             .astype("float32"))
#         score = self.netD(self.netG(z, self.theta_y))
#         loss = -torch.sigmoid(score).mean()
#
#         loss.backward()
#         grad = self.theta_y.grad.cpu().detach().clone().numpy()
#         grad_norm = np.linalg.norm(grad)
#
#         return grad, grad_norm


class BAYES_GAN_penalty(base):
    @auto_init_pytorch
    def __init__(self, gen_init, indices, disc_init, experiment):
        super(BAYES_GAN_penalty, self).__init__(
            gen_init, indices, disc_init, experiment)

    def trainD(self, x, verbose=False, *args, **kwargs):
        score1 = -torch.sigmoid(self.netD(x)).mean()
        z = self.to_var(np.random.normal(
            size=(self.expe.config.ssize, self.expe.config.zsize))
            .astype("float32"))
        score2 = torch.sigmoid(
            self.netD(self.netG(z).detach())).mean()
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
        self.optimize(self.optD, loss)

        return loss

    def get_gradient(self, x, *args, **kwargs):
        z = self.to_var(np.random.normal(
            size=(self.expe.config.ssize, self.expe.config.zsize))
            .astype("float32"))
        score = self.netD(self.netG(z))
        loss = -torch.sigmoid(score).mean()

        loss.backward()
        grad = [x_.grad.cpu().detach().clone().numpy()
                for x_ in self.netG.parameters() if x_.requires_grad]
        grad_norm = np.linalg.norm(grad)
        # grad_maxnorm = np.absolute(grad).max()

        return grad, grad_norm


class BAYES_JS_GAN(base):
    @auto_init_pytorch
    def __init__(self, gen_init, indices, disc_init, experiment):
        super(BAYES_JS_GAN, self).__init__(
            gen_init, indices, disc_init, experiment)

    def trainD(self, x, verbose=False, *args, **kwargs):

        # generate new theta & fake X
        score1 = torch.log(torch.sigmoid(self.netD(self.to_var(x)))).mean()
        z = self.to_var(np.random.normal(
            size=(self.expe.config.ssize, self.expe.config.zsize))
            .astype("float32"))
        score2 = torch.log(1 - torch.sigmoid(
            self.netD(self.netG(z).detach()))).mean()
        loss = - score1 - score2 - np.log(4)
        self.optimize(self.optD, loss)

        # if verbose:
        #     self.expe.log.info(
        #         "optimizing D, loss: {:.4f}".format(loss.item()))
        return loss

    def get_gradient(self, x, *args, **kwargs):
        z = self.to_var(np.random.normal(
            size=(self.expe.config.ssize, self.expe.config.zsize))
            .astype("float32"))
        score = torch.log(1 - torch.sigmoid(
            self.netD(self.netG(z)))).mean()
        loss = score

        loss.backward()
        grad = [x_.grad.cpu().detach().clone().numpy()
                for x_ in self.netG.parameters() if x_.requires_grad]
        grad_norm = np.linalg.norm(grad)

        return grad, grad_norm
