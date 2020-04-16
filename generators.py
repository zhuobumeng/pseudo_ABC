import torch


class trivial(torch.nn.Module):
    def __init__(self, out_unit, data_init, device):
        super(trivial, self).__init__()
        self.out_unit = out_unit
        self.device = device
        self.bias = torch.nn.Parameter(
            data=torch.from_numpy(data_init).to(device=device),
            requires_grad=True)

    def forward(self, invec):
        return self.bias + invec

    def extra_repr(self):
        return 'out_features={}, no bias'.format(self.out_unit)


class bayes_trivial(torch.nn.Module):
    def __init__(self, out_unit, device):
        super(bayes_trivial, self).__init__()
        self.out_unit = out_unit
        self.device = device

    def forward(self, invec, theta):
        return theta + invec

    def extra_repr(self):
        return 'sampling based, no param defined \
            for netG function (param input)'


class gnk(torch.nn.Module):
    def __init__(self, out_unit, params_init, indices, device):
        super(gnk, self).__init__()
        a, b, g, k = params_init
        self.out_unit = out_unit
        self.device = device
        self.a = torch.nn.Parameter(
            data=torch.tensor(a).to(device=device),
            requires_grad=bool(indices[0]))
        self.b = torch.nn.Parameter(
            data=torch.tensor(b).to(device=device),
            requires_grad=bool(indices[1]))
        self.g = torch.nn.Parameter(
            data=torch.tensor(g).to(device=device),
            requires_grad=bool(indices[2]))
        self.k = torch.nn.Parameter(
            data=torch.tensor(k).to(device=device),
            requires_grad=bool(indices[3]))

    def forward(self, invec):
        # not torch format
        # invec: n * dim
        # a, b, g, k: 1 * dim or scalar
        # output: n * dim
        tmp = (1 - torch.exp(-self.g*invec)) / (1 + torch.exp(-self.g*invec))
        return self.a + self.b*(1+0.8 * tmp)*((1+invec**2)**self.k)*invec

    def extra_repr(self):
        return 'out_features={}, no bias'.format(self.out_unit)
