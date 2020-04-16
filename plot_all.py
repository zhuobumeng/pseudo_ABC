import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import config
import train_helper
import bayes_models
import pickle
import numpy as np
from scipy.stats import norm
# from tensorboardX import SummaryWriter
from math import gamma
from scipy.special import gammainc
from itertools import groupby
from scipy.stats import distributions

threshold_option = [0.05, 0.01, 0.008, 0.005, 0.004,
                    0.003, 0.001, 0.0005]

binnum = 60


def calstat(x, w):
    x, w = np.array(x).reshape(-1).astype("float32"), np.array(w).reshape(-1).astype('float32')
    fit_mean = np.average(x, weights=w)
    fit_var = np.average((x-fit_mean)**2, weights=w)
    return fit_mean, np.sqrt(fit_var)


def kernel_weights(weights, gradnorm, threshold, method="kernel"):
    if method.lower() == "reject":
        return (np.array(gradnorm) <= threshold).astype("float32") * weights
    elif method.lower() == "kernel":
        return np.array(weights) * np.exp(
                - np.array(gradnorm) ** 2 / (2 * (threshold**2)))
    else:
        raise ValueError("set method to be 'reject' or 'kernel'.")


def combine_weight(rvs, weights):
    unique_rvs = []
    unique_weights = []
    for key, group in groupby(zip(rvs, weights), lambda x: x[0]):
        unique_rvs.append(key)
        unique_weights.append(sum(member[1] for member in group))
    x, y = zip(*sorted(zip(unique_rvs, unique_weights), key=lambda x: x[0]))
    x, y = np.array(x), np.array(y)
    return x, y/y.sum()


def thmplot_pdf(x, d):
    return 2 * (d/2)**(d/2) / \
        gamma(d/2)*x**(d-1) * np.exp(- d/2 * x**2)


def thmplot_cdf(x, d):
    return gammainc(d/2, d * x**2 / 2)


def kstest_weighted(rvs, cdf, weights=None, mode="approx", unique=False,
                    **kwargs):
    """
    alternative is always two-sided;
    """
    # first groupby same rvs and sum-up weights
    if weights is None:
        weights = np.ones(len(rvs))
    if not unique:
        rvs, weights = combine_weight(rvs, weights)
    cdfvals = cdf(rvs, **kwargs)
    up_weight_sum = weights.cumsum()
    low_weight_sum = np.insert(up_weight_sum[:-1], 0, 0)
    Dplus = (up_weight_sum / weights.sum() - cdfvals).max()
    Dmin = (cdfvals - low_weight_sum / weights.sum()).max()
    D = np.max([Dplus, Dmin])
    N = (weights.sum())**2 / (weights**2).sum()

    if mode == "asymp":
        return D, distributions.kstwobign.sf(D*np.sqrt(N)), N
    if mode == 'approx':
        pval_two = distributions.kstwobign.sf(D*np.sqrt(N))
        if N > 2666 or pval_two > 0.80 - N*0.3/1000.0:
            return D, distributions.kstwobign.sf(D*np.sqrt(N)), N
        else:
            return D, distributions.ksone.sf(D, N)*2, N


def plot_all_pdf(all_weights, all_grad_norms, all_theta, all_grad, theta_y,
                 all_dloss, data_size, isize, filename,
                 candidate_threshold=threshold_option):
    min_gradnorm = min(all_grad_norms)
    t = theta_y - np.concatenate(all_theta)
    x_axis_plot = np.sqrt((t ** 2).sum(axis=1) * data_size / isize)
    fig, ax = plt.subplots(
        nrows=3,
        ncols=len(candidate_threshold),
        figsize=(50, 20)
    )
    x_thm = np.arange(0, 10, 0.001)
    y_thm = thmplot_pdf(x_thm, d=isize)
    for i, ker_h in enumerate(candidate_threshold):
        weight_option = [
            kernel_weights(all_weights, all_grad_norms, ker_h),
            (np.array(all_grad_norms) <= ker_h).astype("float32") * all_weights
        ]
        name_option = ["kerh", "rejction"]
        for j in range(2):
            plot_weight = weight_option[j]
            ax[j, i].hist(
                x_axis_plot,
                bins=binnum,
                weights=plot_weight,
                normed=True,
                label='empirical', alpha=0.5)
            ax[j, i].plot(
                x_thm, y_thm,
                label='theoretical', alpha=0.5)
            ax[j, i].set_title(name_option[j] + ": {:.4f}, \
                min_norm:{:.4f}".format(ker_h, min_gradnorm))
            ax[j, i].set_xlim([0, 6])
            ax[j, i].legend(loc='upper right')

    ax[2, 0].scatter(x_axis_plot, all_grad_norms)
    ax[2, 0].set_title("grad norm (y) vs normalized theta (x)")
    if all_dloss is not None:
        ax[2, 1].scatter(x_axis_plot, all_dloss)
        ax[2, 1].set_title("netD loss (y) vs normalized theta (x)")
    ax[2, 2].hist(
        x_axis_plot,
        bins=binnum,
        normed=True,
        label='empirical', alpha=0.5)
    ax[2, 2].plot(
        x_thm, y_thm,
        label='theoretical', alpha=0.5)
    ax[2, 2].set_title("Proposal vs theoretical")
    ax[2, 2].set_xlim([-0.1, 6])
    print("pdf plot drew")
    fig.savefig(filename, format="png")
    plt.cla()
    plt.close()
    print("plot saved to", filename)


def plot_all_cdf(all_weights, all_grad_norms, all_theta, all_grad, theta_y,
                 all_dloss, data_size, isize, filename,
                 candidate_threshold=threshold_option):

    min_gradnorm = min(all_grad_norms)
    t = theta_y - np.concatenate(all_theta)
    x_axis_plot = np.sqrt((t ** 2).sum(axis=1) * data_size / isize)
    fig, ax = plt.subplots(
        nrows=2,
        ncols=len(candidate_threshold),
        figsize=(50, 20)
    )
    xaxis_thm = np.arange(-0.01, 10, 0.01)
    yaxis_thm = thmplot_cdf(xaxis_thm, d=isize)
    for i, ker_h in enumerate(candidate_threshold):
        weight_option = [
            kernel_weights(all_weights, all_grad_norms, ker_h),
            (np.array(all_grad_norms) <= ker_h).astype("float32") * all_weights
        ]
        name_option = ["kerh", "rejction"]
        for j in range(2):
            plot_weight = weight_option[j]
            group_x, group_weight = combine_weight(x_axis_plot, plot_weight)
            ax[j, i].step(group_x, group_weight.cumsum(), label="empirical", color="black")
            ax[j, i].plot(xaxis_thm, yaxis_thm, color="red", label="theoretical")
            ax[j, i].step(group_x, np.arange(1, len(group_x)+1)/len(group_x), label="proposal", color="blue")
            ax[j, i].set_title(name_option[j] + ": {:.4f},\
                min_norm:{:.4f}".format(ker_h, min_gradnorm))
            ax[j, i].set_xlim([-0.1, 10])
            ax[j, i].legend(loc='lower right')

            test_proposal = kstest_weighted(group_x, cdf=thmplot_cdf, args=[isize], unique=True)
            test_emp = kstest_weighted(group_x, weights=group_weight, cdf=thmplot_cdf, args=[isize], unique=True)
            ax[j, i].set_xlabel("P-val: prop {:.4f}, emp {:.4f}, effsize: {:.1f}".format(
                test_proposal[1],
                test_emp[1],
                test_emp[2]
            ))

    print("cdf plot drew")
    fig.savefig(filename, format="png")
    plt.cla()
    plt.close()
    print("plot saved to", filename)


if __name__ == '__main__':
    dir = "bayes_js_5d/bsize100data_size1000dhsize0ds150eps1e-06err0.005ghsize0isize5lr_d0.001lr_g0.001m10.0m21.0mi11.0mi20.0modelbayes_jsn_epoch20n_iteration3000notent500nt_pertheta15osize1pt0.1ssize3000std0.1t0.01v11.0v21.0wt1zsize5"
    [theta_y, all_theta,
     all_weights, all_grad_norm] = pickle.load(
            open(dir + "/bayes_all+thetas+weights.pkl", "rb"))
    [gradlist, all_loss] = pickle.load(
        open(dir + "/bayes_all+gradlist.pkl", "rb"))
    # gradlist, all_loss = pickle.load(
    #     open(dir + "/bayes_all+gradlist.pkl", "rb")), None
    data_size = 1000
    isize = 5
    all_loss = None
    start = 0.05
    stop = 0.2
    candidate_threshold = np.linspace(start, stop, num=10)

    plot_all_pdf(
        all_weights=all_weights,
        all_grad_norms=all_grad_norm,
        all_theta=all_theta,
        all_grad=gradlist,
        theta_y=theta_y,
        all_dloss=all_loss,
        data_size=data_size,
        isize=isize,
        filename=dir + "/all_plots_pdf.png",
        candidate_threshold=candidate_threshold)

    plot_all_cdf(
        all_weights=all_weights,
        all_grad_norms=all_grad_norm,
        all_theta=all_theta,
        all_grad=gradlist,
        theta_y=theta_y,
        all_dloss=all_loss,
        data_size=data_size,
        isize=isize,
        filename=dir + "/all_plots_cdf.png",
        candidate_threshold=candidate_threshold)
