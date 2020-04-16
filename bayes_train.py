import torch
import config
import train_helper
import bayes_models
import pickle
import numpy as np


def run(e):
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    if e.config.model.lower() == "bayes_tv":
        model_class = bayes_models.BAYES_GAN
    if e.config.model.lower() == "bayes_tv_pen":
        model_class = bayes_models.BAYES_GAN_penalty
    if e.config.model.lower() == "bayes_js":
        model_class = bayes_models.BAYES_JS_GAN

    thetalist, netD_state, m1, data = \
        pickle.load(open(
            e.config.weight_path + "/theta+weights+netD+state-dict.pkl", "rb"))
    # assert m1 == e.config.m1, \
    #     "saved m1: {} != current m1: {}".format(m1, e.config.m1)

    # data = np.random.multivariate_normal(
    #     np.zeros(e.config.isize), np.eye(e.config.isize),
    #     size=e.config.data_size).astype("float32")
    # theta_y = data.mean(0)[None, :]
    theta_y = thetalist.mean(0)  # matrix dimension
    # theta_y = np.array([[0] * e.config.isize]).astype("float32")

    gen_init = theta_y.copy()
    # gen_init = np.array([[0] * e.config.isize]).astype("float32")
    model = model_class(
        input_dim=e.config.isize,
        gen_init=gen_init,
        theta_y=theta_y,
        disc_init=netD_state,
        experiment=e)

    e.log.info(model)

    # if e.config.summarize:
    #     writer = SummaryWriter(e.experiment_dir)

    # for it in range(e.config.n_iteration):
    model.train()
    model.get_theta()
    model.save()

    # pickle.dump([model.valid_theta, model.importance_weights],
    #             open(e.experiment_dir + "/bayes_thetas+weights.pkl", "wb+"))
    pickle.dump([
        theta_y, model.all_theta, model.all_delta,
        model.all_weights, model.all_grad_norm, model.all_loss],
        open(e.experiment_dir + "/bayes_all+thetas+weights.pkl", "wb+"))
    # pickle.dump(
    #     [model.gradlist, model.all_loss],
    #     open(e.experiment_dir + "/bayes_all+gradlist.pkl", "wb+"))
    # plot_all_pdf(
    #     all_weights=model.all_weights,
    #     all_grad_norms=model.all_grad_norm,
    #     all_theta=model.all_theta,
    #     # all_grad=model.gradlist,
    #     theta_y=theta_y,
    #     # all_dloss=model.all_loss,
    #     data_size=e.config.data_size,
    #     isize=e.config.isize,
    #     filename=e.experiment_dir + "/all_plots_pdf.png",
    #     candidate_threshold=np.linspace(0.01, 0.2, num=10))
    # plot_all_cdf(
    #     all_weights=model.all_weights,
    #     all_grad_norms=model.all_grad_norm,
    #     all_theta=model.all_theta,
    #     # all_grad=model.gradlist,
    #     theta_y=theta_y,
    #     # all_dloss=model.all_loss,
    #     data_size=e.config.data_size,
    #     isize=e.config.isize,
    #     filename=e.experiment_dir + "/all_plots_cdf.png",
    #     candidate_threshold=np.linspace(0.01, 0.2, num=10))


if __name__ == '__main__':

    args = config.get_base_parser().parse_args()

    with train_helper.experiment(args, args.save_prefix) as e:
        np.random.seed(e.config.random_seed)
        torch.manual_seed(e.config.random_seed)

        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        e.log.info(args)
        e.log.info("*" * 25 + " ARGS " + "*" * 25)

        run(e)
