import numpy as np

from decorators import auto_init_args


class data_loader:
    @auto_init_args
    def __init__(self, mean1, mean2, var1, var2, mix_weight,
                 dim, size, batch_size):
        if mix_weight[1] == 0:
            self.data = np.random.multivariate_normal(
                np.zeros(dim) + mean1, np.eye(dim) * var1, size=size)\
                .astype("float32")
        elif mix_weight[0] == 0:
            self.data = np.random.multivariate_normal(
                np.zeros(dim) + mean2, np.eye(dim) * var2, size=size)\
                .astype("float32")
        else:
            self.data1 = np.random.multivariate_normal(
                np.zeros(dim) + mean1, np.eye(dim) * var1, size=size)\
                .astype("float32")
            self.data2 = np.random.multivariate_normal(
                np.zeros(dim) + mean2, np.eye(dim) * var2, size=size)\
                .astype("float32")
            sum_of_mix = sum(mix_weight)
            mix = np.random.multinomial(
                1, [m / sum_of_mix for m in mix_weight], size=size).T
            self.data = (np.stack([self.data1, self.data2]) * mix[:, :, None])\
                .sum(0).astype("float32")

    def get_mean(self):
        return self.data.mean(0)[None, :]

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def prepare(self):
        self.data = self.data[np.random.permutation(len(self.data))]
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i: i + self.batch_size]

    def sample(self, sample_size=None):
        ssize = self.batch_size if sample_size is None else sample_size
        return self.data[np.random.permutation(len(self.data))
                         [:ssize]]


class gnk_data_loader:
    @auto_init_args
    def __init__(self, a, b, g, k, dim, size, batch_size):
        norm_data = np.random.multivariate_normal(
            np.zeros(dim), np.eye(dim),
            size=size).astype("float32")
        tmp = (1 - np.exp(-g*norm_data)) / (1 + np.exp(-g*norm_data))
        self.data = (a + b*(1+0.8 * tmp)*(
            (1+norm_data**2)**k)*norm_data).astype("float32")

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def prepare(self):
        self.data = self.data[np.random.permutation(len(self.data))]
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i: i + self.batch_size]

    def sample(self, sample_size=None):
        ssize = self.batch_size if sample_size is None else sample_size
        return self.data[np.random.permutation(len(self.data))
                         [:ssize]]
