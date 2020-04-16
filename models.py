import os
import torch
import generators

import numpy as np
import torch.nn as nn

from decorators import auto_init_pytorch


class base(nn.Module):
    def __init__(self, input_dim, data_init, experiment):
        super(base, self).__init__()
        self.expe = experiment
        self.eps = self.expe.config.eps
        self.expe.log.info(
            "norm of data mean: {:.4f}".format(
                ((data_init ** 2).sum() ** 0.5).item()))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if self.expe.config.dhsize:
            self.netD = nn.Sequential(
                nn.Linear(input_dim, self.expe.config.dhsize),
                nn.Tanh(),
                nn.Linear(self.expe.config.dhsize, self.expe.config.osize))
        else:
            # self.netD = nn.Sequential(
                # nn.Linear(input_dim, self.expe.config.osize, bias=False))
            self.netD = nn.Sequential(
                nn.Linear(input_dim, self.expe.config.osize))
        self.netD = self.netD.to(self.device)

        if self.expe.config.ghsize:
            self.netG = nn.Sequential(
                nn.Linear(self.expe.config.zsize, self.expe.config.ghsize),
                nn.Tanh(),
                nn.Linear(self.expe.config.ghsize, input_dim)
            )
        else:
            self.netG = generators.trivial(
                input_dim, data_init, self.device)

        self.theta = [self.netG.bias.detach().clone().cpu().numpy()]
        self.thetalist = [self.netG.bias.detach().clone().cpu().numpy()]

    def update_theta(self):
        new_theta = self.netG.bias.detach().clone().cpu().numpy()
        self.thetalist.append(new_theta)
        if len(self.theta) < self.expe.config.nt:
            self.theta.append(new_theta)
        else:
            self.theta = self.theta[1:] + [new_theta]

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
            "theta": self.theta,
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
        z = self.to_var(
            np.zeros((1, self.expe.config.zsize)).astype("float32"))
        point_est = (self.netG(z) - self.expe.config.m1)\
            .pow(2).sum().pow(0.5).item()

        span_est = (((np.stack(self.theta).mean(0) -
                      self.expe.config.m1) ** 2).sum() ** 0.5).item()

        all_est = (((np.stack(self.thetalist).mean(0) -
                     self.expe.config.m1) ** 2).sum() ** 0.5).item()

        return point_est, span_est, all_est

    # get variance of T
    def get_varT(self):
        z1 = self.to_var(np.random.normal(
            size=(self.expe.config.ssize, self.expe.config.zsize))
            .astype("float32"))

        z2 = self.to_var(np.random.normal(
            size=(self.expe.config.ssize, self.expe.config.zsize))
            .astype("float32"))
        left = torch.sigmoid(self.netD(self.netG(z1))).mean()
        right = (torch.sigmoid(self.netD(self.netG(z2))) - left).pow(2).mean()

        return right

    def get_varT_data(self, x):
        x = self.to_var(x)
        left = self.netD(self.netG(x)).mean()
        right = (self.netD(self.netG(x)) - left).pow(2).mean()

        return right


class JS_GAN(base):
    @auto_init_pytorch
    def __init__(self, input_dim, data_init, experiment):
        assert experiment.config.osize == 1, \
            "{} only supports ouput size 1".format(type(self))
        super(JS_GAN, self).__init__(input_dim, data_init, experiment)

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
        loss = - score1 - score2 - np.log(4)
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
    def __init__(self, input_dim, data_init, experiment):
        assert experiment.config.osize == 1, \
            "{} only supports ouput size 1".format(type(self))
        super(TV_GAN, self).__init__(input_dim, data_init, experiment)

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
    def __init__(self, input_dim, data_init, experiment):
        super(TV_GAN_penalty, self).__init__(input_dim, data_init, experiment)

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


class FREQ_GAN(base):
    @auto_init_pytorch
    def __init__(self, input_dim, data_init, experiment):
        super(FREQ_GAN, self).__init__(input_dim, data_init, experiment)

    def trainD(self, x, *args, **kwargs):
        varT = self.get_varT()
        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = torch.sigmoid(self.netD(self.to_var(x))).mean()
        score2 = torch.sigmoid(self.netD(self.netG(z).detach())).mean()
        loss = (score1 - score2).pow(2) / varT
        loss = loss + torch.log(varT) / self.expe.config.data_size

        loss = -loss
        self.optimize(self.optD, loss)
        return loss

    def trainG(self, x, *args, **kwargs):
        varT = self.get_varT()

        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = torch.sigmoid(self.netD(self.to_var(x))).mean()
        score2 = torch.sigmoid(self.netD(self.netG(z))).mean()
        loss = (score1 - score2).pow(2) / varT
        loss = loss + torch.log(varT) / self.expe.config.data_size

        self.optimize(self.optG, loss)
        return loss


class FREQB_GAN(base):
    @auto_init_pytorch
    def __init__(self, input_dim, data_init, experiment):
        super(FREQB_GAN, self).__init__(input_dim, data_init, experiment)

    def trainD(self, x, *args, **kwargs):
        varT = self.get_varT()
        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = torch.sigmoid(self.netD(self.to_var(x))).mean()
        score2 = torch.sigmoid(self.netD(self.netG(z).detach())).mean()
        loss = (score1 - score2) / (varT + self.eps).pow(0.5)
        # loss = loss + torch.log(varT) / self.expe.config.data_size

        loss = -loss
        self.optimize(self.optD, loss)
        return loss

    def trainG(self, x, *args, **kwargs):
        varT = self.get_varT()

        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = torch.sigmoid(self.netD(self.to_var(x))).mean()
        score2 = torch.sigmoid(self.netD(self.netG(z))).mean()
        loss = (score1 - score2) / (varT + self.eps).pow(0.5)

        self.optimize(self.optG, loss)
        return loss


# weird FREQC
class FREQC_GAN(base):
    @auto_init_pytorch
    def __init__(self, input_dim, data_init, experiment):
        super(FREQC_GAN, self).__init__(input_dim, data_init, experiment)

    def trainD(self, x, *args, **kwargs):
        varT = self.get_varT()
        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = torch.sigmoid(self.netD(self.to_var(x))).mean()
        score2 = torch.sigmoid(self.netD(self.netG(z).detach())).mean()
        loss = torch.abs(score1 - score2) / (varT + self.eps).pow(0.5)
        loss = - loss
        self.optimize(self.optD, loss)
        return loss

    def trainG(self, x, *args, **kwargs):
        varT = self.get_varT()

        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = torch.sigmoid(self.netD(self.to_var(x))).mean()
        score2 = torch.sigmoid(self.netD(self.netG(z))).mean()
        loss = torch.abs(score1 - score2) / (varT + self.eps).pow(0.5)

        self.optimize(self.optG, loss)
        return loss


# fix at truth only update D
class FREQD_GAN(base):
    @auto_init_pytorch
    def __init__(self, input_dim, data_init, experiment):
        data_init = np.array(
            [
                [experiment.config.m1] * experiment.config.isize
            ]).astype("float32")
        super(FREQD_GAN, self).__init__(input_dim, data_init, experiment)
        # print("data_init", data_init)

    def trainD(self, x, *args, **kwargs):
        varT = self.get_varT()
        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = torch.sigmoid(self.netD(self.to_var(x))).mean()
        score2 = torch.sigmoid(self.netD(self.netG(z).detach())).mean()
        loss = (score1 - score2) / (varT + self.eps).pow(0.5)
        # loss = loss + torch.log(varT) / self.expe.config.data_size

        loss = -loss
        self.optimize(self.optD, loss)
        return loss * (varT + self.eps).pow(0.5)

    def trainG(self, x, *args, **kwargs):
        varT = self.get_varT()

        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = torch.sigmoid(self.netD(self.to_var(x))).mean()
        score2 = torch.sigmoid(self.netD(self.netG(z))).mean()
        loss = (score1 - score2) / (varT + self.eps).pow(0.5)

        # self.optimize(self.optG, loss)
        return loss


# Squared version
class FREQE_GAN(base):
    @auto_init_pytorch
    def __init__(self, input_dim, data_init, experiment):
        super(FREQE_GAN, self).__init__(input_dim, data_init, experiment)

    def trainD(self, x, *args, **kwargs):
        varT = self.get_varT_data(x)
        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = torch.sigmoid(self.netD(self.to_var(x))).mean()
        score2 = torch.sigmoid(self.netD(self.netG(z).detach())).mean()
        loss = (score1 - score2).pow(2) / (varT + self.eps)
        loss = - loss
        self.optimize(self.optD, loss)
        return loss

    def trainG(self, x, *args, **kwargs):
        varT = self.get_varT_data(x)

        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = torch.sigmoid(self.netD(self.to_var(x))).mean()
        score2 = torch.sigmoid(self.netD(self.netG(z))).mean()
        loss = (score1 - score2).pow(2) / (varT + self.eps)

        self.optimize(self.optG, loss)
        return loss


# exponential expression
class FREQF_GAN(base):
    """based on freqB but added iterative update varT"""
    @auto_init_pytorch
    def __init__(self, input_dim, data_init, experiment):
        super(FREQF_GAN, self).__init__(input_dim, data_init, experiment)

    def trainD(self, x, *args, **kwargs):
        varT = self.get_varT()
        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = torch.sigmoid(self.netD(self.to_var(x))).mean()
        score2 = torch.sigmoid(self.netD(self.netG(z).detach())).mean()
        score = (1 / varT).pow(0.5) * torch.exp(
            - len(x) * (score1 - score2).pow(2) / varT)
        # loss = loss + torch.log(varT) / self.expe.config.data_size

        loss = score
        self.optimize(self.optD, loss)
        return loss

    def trainG(self, x, *args, **kwargs):
        varT = self.get_varT()

        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = torch.sigmoid(self.netD(self.to_var(x))).mean()
        score2 = torch.sigmoid(self.netD(self.netG(z))).mean()
        score = (1 / varT).pow(0.5) * torch.exp(
            - len(x) * (score1 - score2).pow(2) / varT)
        loss = - score

        self.optimize(self.optG, loss)
        return loss


# Add one more param
class FREQG_GAN(base):
    """FreqB + variance regularization"""
    @auto_init_pytorch
    def __init__(self, input_dim, data_init, experiment):
        super(FREQG_GAN, self).__init__(input_dim, data_init, experiment)

    def trainD(self, x, *args, **kwargs):
        varT = self.get_varT()
        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = self.netD(self.to_var(x)).mean()
        score2 = self.netD(self.netG(z).detach()).mean()
        loss = (score1 - score2) / (varT + self.eps).pow(0.5)

        loss = -loss
        self.optimize(self.optD, loss)
        return loss

    def trainG(self, x, *args, **kwargs):
        varT = self.get_varT()

        z = self.to_var(np.random.normal(
            size=(len(x), self.expe.config.zsize)).astype("float32"))
        score1 = self.netD(self.to_var(x)).mean()
        score2 = self.netD(self.netG(z).detach()).mean()
        loss = (score1 - score2) / (varT + self.eps).pow(0.5)

        self.optimize(self.optG, loss)
        return loss
