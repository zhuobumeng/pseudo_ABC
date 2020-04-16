import os
import torch
import generators

import numpy as np
import torch.nn as nn

from decorators import auto_init_pytorch


class base(nn.Module):
    def __init__(self, params_init, indices, experiment):
        super(base, self).__init__()
        self.expe = experiment
        self.eps = self.expe.config.eps
        input_dim = self.expe.config.isize
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        if self.expe.config.dhsize:
            self.netD = nn.Sequential(
                nn.Linear(input_dim, self.expe.config.dhsize),
                getattr(nn, self.expe.config.activate)(),
                nn.Linear(self.expe.config.dhsize, self.expe.config.osize))
        else:
            self.netD = nn.Sequential(
                nn.Linear(input_dim, self.expe.config.osize))
        self.netD = self.netD.to(self.device)

        self.netG = generators.gnk(
            input_dim, params_init, indices, self.device)
        # self.theta = [self.netG.bias.detach().clone().cpu().numpy()]
        self.thetalist = []

    def update_theta(self):
        # new_theta = self.netG.bias.detach().clone().cpu().numpy()
        new_theta = [x_.detach().clone().cpu().numpy()
                     for x_ in self.netG.parameters()]
        self.thetalist.append(new_theta)
        # if len(self.theta) < self.expe.config.nt:
        #     self.theta.append(new_theta)
        # else:
        #     self.theta = self.theta[1:] + [new_theta]

    def measure_grad(self, params, indices):
        _netG = generators.gnk(self.expe.config.isize, params,
                               indices, self.device)
        z = self.to_var(np.random.normal(
                size=(self.expe.config.ssize, self.expe.config.zsize)
            ).astype("float32"))
        score = self.netD(_netG(z))
        loss = -torch.sigmoid(score).mean()
        self.zero_grad()
        loss.backward()
        grad = [x_.grad.cpu().detach().clone().numpy()
                for x_ in _netG.parameters() if x_.requires_grad]
        grad_norm = np.linalg.norm(grad)
        return grad, grad_norm

    def to_var(self, inputs):
        if torch.is_tensor(inputs):
            return inputs.clone().detach().to(device=self.device)
        else:
            return torch.tensor(inputs, device=self.device)

    def to_vars(self, *inputs):
        return [self.to_var(inputs_) if inputs_ is not None and inputs_.size
                else None for inputs_ in inputs]

    # update one gradient step
    def optimize(self, opt, loss):
        self.zero_grad()
        loss.backward()
        opt.step()
        # for p in self.netD.parameters():
        #     p.data.clamp_(-self.expe.config.wt, self.expe.config.wt)

    # define optimizer
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
            "thetalist": self.thetalist,
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

    def eval_error(self):
        # true_param = np.array([
        #     self.expe.config.a, self.expe.config.b,
        #     self.expe.config.g, self.expe.config.k]
        # ).astype('float32')
        true_param = np.array(self.expe.config.params).astype("float32")
        curr_theta = np.array([
            x_.detach().clone().cpu().numpy()
            for x_ in self.netG.parameters()]
        ).astype("float32")
        point_est = (((curr_theta - true_param)**2).sum()**0.5).item()
        all_est = (((np.stack(self.thetalist).mean(0)
                     - true_param) ** 2).sum() ** 0.5).item()
        return point_est, all_est

    # get variance of T
    # def get_varT(self):
    #     z1 = self.to_var(np.random.normal(
    #         size=(self.expe.config.ssize, self.expe.config.zsize))
    #         .astype("float32"))
    #     z2 = self.to_var(np.random.normal(
    #         size=(self.expe.config.ssize, self.expe.config.zsize))
    #         .astype("float32"))
    #     left = torch.sigmoid(self.netD(self.netG(z1))).mean()
    #     right = (torch.sigmoid(self.netD(self.netG(z2))) - left).pow(2).mean()
    #     return right

    # def get_varT_data(self, x):
    #     x = self.to_var(x)
    #     left = self.netD(self.netG(x)).mean()
    #     right = (self.netD(self.netG(x)) - left).pow(2).mean()
    #     return right


class JS_GAN(base):
    @auto_init_pytorch
    def __init__(self, params_init, indices, experiment):
        assert experiment.config.osize == 1, \
            "{} only supports ouput size 1".format(type(self))
        super(JS_GAN, self).__init__(params_init, indices, experiment)

    def trainD(self, x, *args, **kwargs):
        # score = self.netD(self.to_var(x)).squeeze(-1)
        # loss1 = F.binary_cross_entropy_with_logits(
        #     score, torch.ones_like(score))
        score1 = torch.log(torch.sigmoid(self.netD(self.to_var(x)))).mean()

        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score2 = torch.log(1 - torch.sigmoid(
            self.netD(self.netG(z).detach()))).mean()
        # loss2 = F.binary_cross_entropy_with_logits(
        #     score, torch.zeros_like(score))
        loss = - score1 - score2
        self.optimize(self.optD, loss)
        return loss

    def trainG(self, x, *args, **kwargs):
        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score = torch.log(1 - torch.sigmoid(self.netD(self.netG(z)))).mean()
        # loss = F.binary_cross_entropy_with_logits(
        #           score, torch.ones_like(score))
        loss = score
        self.optimize(self.optG, loss)
        return loss


class TV_GAN(base):
    @auto_init_pytorch
    def __init__(self, params_init, indices, experiment):
        assert experiment.config.osize == 1, \
            "{} only supports ouput size 1".format(type(self))
        super(TV_GAN, self).__init__(params_init, indices, experiment)

    def trainD(self, x, *args, **kwargs):
        score = self.netD(self.to_var(x)).squeeze(-1)
        loss1 = -torch.sigmoid(score).mean()

        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score = self.netD(self.netG(z).detach())

        loss2 = torch.sigmoid(score).mean()
        loss = loss1 + loss2
        self.optimize(self.optD, loss)

        return loss

    def trainG(self, x, *args, **kwargs):
        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score = self.netD(self.netG(z))
        # loss = - score.mean()
        loss = - torch.sigmoid(score).mean()
        self.optimize(self.optG, loss)
        return loss


class TV_GAN_penalty(base):
    @auto_init_pytorch
    def __init__(self, params_init, indices, experiment):
        super(TV_GAN_penalty, self).__init__(params_init, indices, experiment)

    def trainD(self, x, *args, **kwargs):
        score = self.netD(self.to_var(x)).squeeze(-1)
        loss1 = -torch.sigmoid(score).mean()

        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score = self.netD(self.netG(z).detach())

        loss2 = torch.sigmoid(score).mean()
        loss = loss1 + loss2

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

    def trainG(self, x, *args, **kwargs):
        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score = self.netD(self.netG(z))
        # loss = - score.mean()
        loss = - torch.sigmoid(score).mean()
        self.optimize(self.optG, loss)
        return loss
