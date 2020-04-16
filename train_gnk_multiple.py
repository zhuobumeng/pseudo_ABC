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

    tot_it = curr_dstep = 0

    model.train()
    for epoch in range(e.config.n_epoch):
        data_batch = data.prepare()  # generator a list
        for it, d in enumerate(data_batch):
            tot_it += 1
            if curr_dstep < e.config.ds:
                model.trainD(d)
                curr_dstep += 1
            else:
                model.trainG(d)
                curr_dstep = 0
                model.update_theta()

    return np.array(model.thetalist).mean(0).astype("float32")


if __name__ == '__main__':
    args = config.get_base_parser().parse_args()

    with train_helper.experiment(args, args.save_prefix) as e:
        np.random.seed(e.config.random_seed)
        torch.manual_seed(e.config.random_seed)

        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        e.log.info(args)
        e.log.info("*" * 25 + " ARGS " + "*" * 25)

        all_theta_list = []
        for i in range(e.config.num_run):
            theta_ = run(e)
            all_theta_list.append(theta_)
            e.log.info("finished {}: theta_list = {}".format(i, theta_))
        pickle.dump(all_theta_list,
                    open(e.experiment_dir + "/all_thetas.pkl", "wb+"))
