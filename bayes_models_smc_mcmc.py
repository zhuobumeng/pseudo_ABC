import os
import torch
import generators

import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
import torch.nn as nn

from decorators import auto_init_pytorch, auto_init_args
# from scipy.stats import multivariate_normal as MN


# class data_loader_edit:
#     @auto_init_args
#     def __init__(self, data, batch_size=100):
#         self.data = data[np.random.permutation(len(data))]

#     def __len__(self):
#         return len(self.data) // self.batch_size + 1

#     def prepare(self):
#         for i in range(0, len(self.data), self.batch_size):
#             yield self.data[i: i + self.batch_size]


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

        # update D state, set initial as the output of freq
        self.netD = self.netD.to(self.device)
        # self.netD.load_state_dict(disc_init)

        self.gen_init = gen_init  # Start initial center
        # self.gen_init = self.to_var(gen_init)
        self.theta_y_numpy = theta_y  # Freq best est
        self.theta_y = nn.Parameter(self.to_var(theta_y))
        self.theta_y.requires_grad = True

        # Set Generator Model
        self.netG = generators.bayes_trivial(input_dim, self.device)

        # Parameter of interest

        # We always keep delta: which is normed theta
        self.all_delta = []  # np.empty(shape=(0, input_dim)).astype("float32")
        self.all_theta = []
        self.all_weights = []
        self.all_grad_norm = []
        self.all_loss = []
        self.all_tolerance = []

        # self.gen_mean = np.zeros((1, input_dim)).astype("float32")
        # self.gen_std = np.ones((1, input_dim)).astype("float32") * 3

    # def compute_weights(self, delta, gen_mean, gen_std):

    #     scale_delta = (delta - gen_mean) / gen_std
    #     theta_numpy = self.theta_y_numpy - delta / \
    #         np.sqrt(self.expe.config.data_size)
    #     n = -0.5 * (theta_numpy ** 2).sum() + 0.5 * (scale_delta ** 2).sum()
    #     # print("n before", n)
    #     n = n - self.expe.config.isize / 2 * np.log(self.expe.config.data_size)
    #     n = n + 1/2 * (np.log(gen_std)).sum()
    #     # print("n after", n)
    #     # print("*" * 5)
    #     return np.exp(n)

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

    def find_eps(self, eps, grad_norm, delta):
        _, indices = np.unique(delta, return_index=True)
        grad_norm_unique = np.array(grad_norm)[indices]
        self.expe.log.info("Unique sample size: " + str(len(grad_norm_unique)))
        tmp = (grad_norm_unique < eps).sum()
        if tmp <= self.expe.config.total_sample * self.expe.config.eff_prop:
            self.expe.log.info("eps too small, not 1/2 expected, \
                return eps to next time")
            return eps
        else:
            self.expe.log.info("return eps that makes half")
            return np.quantile(grad_norm_unique, self.expe.config.eff_prop)

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
        return grad_norm, loss

    # def theta_2_grad(self, theta):
    #     z = self.to_var(np.random.normal(
    #         size=(self.expe.config.data_size, self.expe.config.zsize))
    #         .astype("float32"))
    #     fake_x = data_loader_edit(
    #         self.netG(z, self.to_var(theta)),
    #         self.expe.config.bsize)
    #     fake_x_batch = fake_x.prepare()
    #     grad, loss = 0, 0
    #     for _, fake_data in enumerate(fake_x_batch):
    #         for ds_ in range(self.expe.config.ds):
    #             loss_ = self.trainD(fake_data, verbose=False)
    #         # only save the final step result for each data batch
    #         loss = loss + loss_ / len(fake_x)
    #         self.zero_grad()
    #         grad_ = self.get_gradient(fake_data)
    #         grad = grad + (grad_ / len(fake_x)).astype("float32")

    #     grad_norm = np.linalg.norm(grad)
    #     return grad_norm, loss

    def get_theta(self, verbose=False):
        curr_delta = []
        curr_grad_norm = []
        curr_loss = []
        curr_theta = []
        curr_eps = np.infty
        # round 0
        for _ in range(self.expe.config.total_sample):
            # new_delta = (np.random.uniform(
            #     -10, 10, size=self.gen_init.shape).astype("float32"))
            new_delta = np.random.normal(size=self.gen_init.size())
            new_theta = self.gen_init + new_delta / np.sqrt(
                self.expe.config.data_size)
            # for _ in range(self.expe.config.nt_pertheta):
            grad_norm, loss = self.theta_2_grad(new_theta)
            curr_loss.append(loss)
            curr_grad_norm.append(grad_norm)
            curr_delta.append(new_delta)
            curr_theta.append(new_theta)
        self.all_grad_norm.append(curr_grad_norm)
        self.all_loss.append(curr_loss)
        self.all_delta.append(curr_delta)
        self.all_theta.append(curr_theta)
        self.all_tolerance.append(curr_eps)

        for t in range(self.expe.config.smc_T):
            next_eps = self.find_eps(curr_eps, curr_grad_norm, curr_delta)
            self.expe.log.info("time:" + str(t) + " eps:" + str(next_eps))
            # if next_eps == -1:
            #     self.expe.log.info("Stop at time" + str(t))
            #     break
            rnd_weights = (
                np.array(curr_grad_norm) <= next_eps).astype("float32")
            self.expe.log.info("Kept sample size: " + str(rnd_weights.sum()))
            rnd_weights = rnd_weights / rnd_weights.sum()
            assert self.theta_y_numpy.shape == new_delta.shape, \
                "theta_y shape not match"
            last_covariance = self.emperical_variance(
                np.concatenate(curr_delta), rnd_weights)
            self.expe.log.info("covariance" + str(last_covariance))
            # self.expe.log.info("shape:" + str(last_covariance.shape))
            # exit()
            half_cov = sqrtm(last_covariance)
            inv_cov = inv(last_covariance)

            curr_delta = []
            curr_grad_norm = []
            curr_loss = []
            curr_theta = []
            for _ in range(self.expe.config.total_sample):
                next_delta, next_theta, loss, grad_norm = self.MCMCkernel(
                    next_eps, rnd_weights,
                    self.all_delta[-1], self.all_theta[-1],
                    self.all_grad_norm[-1], self.all_loss[-1],
                    half_cov, inv_cov)
                curr_loss.append(loss)
                curr_grad_norm.append(grad_norm)
                curr_delta.append(new_delta)
                curr_theta.append(new_theta)
            curr_eps = next_eps
            self.all_grad_norm.append(curr_grad_norm)
            self.all_loss.append(curr_loss)
            self.all_delta.append(curr_delta)
            self.all_theta.append(curr_theta)
            self.all_tolerance.append(curr_eps)

    def emperical_variance(self, x, w):
        # x shape: size * dim, w: size
        return np.dot(x.T * w.reshape((1, -1)), x)

    def MCMCkernel(self, eps, rnd_weights, last_delta, last_theta,
                   last_grad_norm, last_loss, half_cov, inv_cov):
        idx = np.random.choice(
            np.arange(len(last_delta)), p=rnd_weights.flatten())
        hit = 0
        count_hit = [0, 0, 0]
        while True:
            if sum(count_hit) % 200 == 0:
                self.expe.log.info("count_hit " + str(count_hit))
            randomness = np.random.normal(
                size=self.theta_y_numpy.shape).astype("float32")
            new_delta = np.dot(randomness, half_cov)
            new_theta = self.gen_init + new_delta / np.sqrt(
                self.expe.config.data_size)
            grad_norm, loss = self.theta_2_grad(new_theta)
            if grad_norm <= eps:
                hit = hit + 1
                if hit == 1:
                    possible_delta, possible_theta = new_delta, new_theta
                    possible_grad_norm, possible_loss = grad_norm, loss
                if hit == 3:
                    break
            count_hit[hit] = count_hit[hit] + 1

        prob_ratio = np.exp(-0.5 * self.diff_weighted(
            last_delta[idx],
            possible_delta, inv_cov)
        ) * count_hit[2] / (count_hit[0] + count_hit[1])
        if np.random.uniform() <= prob_ratio:
            return possible_delta, possible_theta, possible_loss, possible_grad_norm
        else:
            return last_delta[idx], last_theta[idx], last_loss[idx], last_grad_norm[idx]

    def diff_weighted(self, a, b, inv_cov):
        # a: 1*dim, b: 1*dim, invcov: dim*dim
        return (np.dot(np.dot((a - b), inv_cov), (a + b).T)).item()


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
        return loss.item()

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

        return loss.item()

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
        return loss.item()

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
