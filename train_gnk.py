import torch
import config
import data_utils
import train_helper
import models_gnk as models
import pickle
import numpy as np


def run(e):
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)
    if e.config.model.lower() == "js":
        model_class = models.JS_GAN
    elif e.config.model.lower() == "tv":
        model_class = models.TV_GAN
    elif e.config.model.lower() == "tv_pen":
        model_class = models.TV_GAN_penalty

    [a, b, g, k] = e.config.params
    indices = np.array(e.config.params) != np.array(e.config.params_init)
    data = data_utils.gnk_data_loader(
        a=a, b=b, g=g, k=k,
        dim=e.config.isize,
        size=e.config.data_size,
        batch_size=e.config.bsize)

    model = model_class(
        params_init=e.config.params_init,
        indices=indices,
        experiment=e)

    e.log.info(model)

    # if e.config.summarize:
    #     writer = SummaryWriter(e.experiment_dir)

    # train_stats = train_helper.tracker(
    #     ["discriminator_loss", "generator_loss"])

    tot_it = curr_dstep = 0
    model.train()
    for epoch in range(e.config.n_epoch):
        data_batch = data.prepare()  # generator a list
        for it, d in enumerate(data_batch):
            tot_it += 1
            if curr_dstep < e.config.ds:
                dloss = model.trainD(d)
                curr_dstep += 1
                # train_stats.update(
                #     {"discriminator_loss": dloss},
                #     len(d))
            else:
                gloss = model.trainG(d)
                # train_stats.update(
                #     {"generator_loss": gloss},
                #     len(d))
                curr_dstep = 0
                model.update_theta()

            # if tot_it % e.config.print_every == 0 or \
            #         tot_it % len(data) == 0:
                # summarization = train_stats.summarize(
                #     "epoch: {}, it: {} (max: {})".format(epoch, it, len(data)))
                # e.log.info(summarization)
                # if e.config.summarize:
                #     writer.add_scalar(
                #         "loss/discriminator",
                #         train_stats['discriminator_loss'], tot_it)
                #     writer.add_scalar(
                #         "loss/generator",
                #         train_stats['generator_loss'], tot_it)
                # train_stats.reset()

            if tot_it % e.config.eval_every == 0 or tot_it % len(data) == 0:
                model.eval()
                # with torch.no_grad():
                #     point_err, all_err = \
                #         model.eval_error()
                #     varT = model.get_varT().item()
                # e.log.info("point square_error: {:.3f}, "
                #            "all square error:{:.3f}"
                #            .format(point_err, all_err))
                # if e.config.summarize:
                #     writer.add_scalar(
                #         "square_error/point", point_err, tot_it)
                #     # writer.add_scalar(
                #     #     "square_error/span", span_err, tot_it)
                #     writer.add_scalar(
                #         "square_error/all", all_err, tot_it)
                #     # writer.add_scalar(
                #     #     "varT", varT, tot_it)
                #     for nlayer, layer in enumerate(model.netD):
                #         if hasattr(layer, "weight"):
                #             writer.add_scalar(
                #                 "netD_weight/layer{}"
                #                 .format(nlayer),
                #                 np.linalg.norm(layer.weight.detach().clone().cpu().numpy()).item(),
                #                 tot_it)
                # if e.config.dhsize == 0:

                # all_weights.append(
                #     model.netD[-1].weight.detach().clone().cpu().numpy())
                # train_helper.plot_theta_trajectory(
                #     all_weights, [e.config.m1, e.config.m2],
                #     [e.config.m1, e.config.m2],
                #     e.experiment_dir + "/weight_traj_plot.png")

                # train_helper.plot_theta_trajectory(
                #     model.thetalist, [e.config.m1, e.config.m2],
                #     [e.config.m1, e.config.m2],
                #     e.experiment_dir + "/theta_traj_plot.png")
                # train_stats.reset()
        if epoch % 50 == 0 or epoch == e.config.n_epoch - 1:
            _params = np.array(model.thetalist).mean(0).tolist()
            # _grad, _ = model.measure_grad(_params, indices)
            e.log.info("est: " + str(_params))
    model.save()

    # train_helper.plot_theta_trajectory(
    #     model.thetalist, [e.config.m1, e.config.m2],
    #     [e.config.m1, e.config.m2],
    #     e.experiment_dir + "/theta_traj_plot.png")
    # if e.config.dhsize == 0:
    #     print(all_weights, len(all_weights))
    #     train_helper.plot_theta_trajectory(
    #         all_weights, [e.config.m1, e.config.m2],
    #         [e.config.m1, e.config.m2],
    #         e.experiment_dir + "/weight_traj_plot.png")
    # pickle.dump(data.get_mean(), open(e.experiment_dir + "/data_mean.pkl", "wb+"))
    # pickle.dump(model.thetalist, open(e.experiment_dir + "/all_thetas.pkl", "wb+"))
    pickle.dump([model.thetalist, model.netD.state_dict(),
                 e.config.params, indices],
                open(e.experiment_dir + "/theta+netD+state-dict.pkl", "wb+"))


if __name__ == '__main__':
    args = config.get_base_parser().parse_args()

    with train_helper.experiment(args, args.save_prefix) as e:
        np.random.seed(e.config.random_seed)
        torch.manual_seed(e.config.random_seed)

        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        e.log.info(args)
        e.log.info("*" * 25 + " ARGS " + "*" * 25)

        run(e)
