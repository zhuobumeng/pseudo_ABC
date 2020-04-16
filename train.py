import torch
import config
import data_utils
import train_helper
import models
import pickle

import numpy as np

# from tensorboardX import SummaryWriter


def run(e):
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    if e.config.model.lower() == "js":
        model_class = models.JS_GAN
    elif e.config.model.lower() == "tv":
        model_class = models.TV_GAN
    elif e.config.model.lower() == "tv_pen":
        model_class = models.TV_GAN_penalty
    elif e.config.model.lower() == "freq":
        model_class = models.FREQ_GAN
    elif e.config.model.lower() == "freqb":
        model_class = models.FREQB_GAN
    elif e.config.model.lower() == "freqc":
        model_class = models.FREQC_GAN
    elif e.config.model.lower() == "freqc2":
        model_class = models.FREQC2_GAN
    elif e.config.model.lower() == "freqd":
        model_class = models.FREQD_GAN
    elif e.config.model.lower() == "freqe":
        model_class = models.FREQE_GAN
    elif e.config.model.lower() == "freqf":
        model_class = models.FREQF_GAN
    elif e.config.model.lower() == "freqg":
        model_class = models.FREQG_GAN

    data = data_utils.data_loader(
        mean1=e.config.m1,
        var1=e.config.v1,
        mean2=e.config.m2,
        var2=e.config.v2,
        mix_weight=[e.config.mi1, e.config.mi2],
        dim=e.config.isize,
        size=e.config.data_size,
        batch_size=e.config.bsize)
    # data = data_utils.data_loader(
    #     mean1=e.config.m1,
    #     var1=e.config.v1,
    #     dim=e.config.isize,
    #     size=e.config.data_size,
    #     batch_size=e.config.bsize)
    e.log.info("data mean: " + str(data.data.mean(0)))

    model = model_class(
        input_dim=e.config.isize,
        # data_init=np.array([[0, 1]]).astype("float32"),
        data_init=(
            data.get_mean() + 1 / np.sqrt(e.config.isize)).astype('float32'),
        experiment=e)

    e.log.info(model)

    tot_it = curr_dstep = 0

    for epoch in range(e.config.n_epoch):
        data_batch = data.prepare()  # generator a list
        for it, d in enumerate(data_batch):
            model.train()
            tot_it += 1

            if curr_dstep < e.config.ds:
                dloss = model.trainD(d)
                curr_dstep += 1
            else:
                gloss = model.trainG(d)
                curr_dstep = 0
                model.update_theta()

            if tot_it % e.config.eval_every == 0 or tot_it % len(data) == 0:
                model.eval()

    model.save()
    e.log.info("estimate: "+ str(np.stack(model.thetalist).mean(0)))
    return [np.stack(model.thetalist), model.netD.state_dict(), e.config.m1, data.data]


if __name__ == '__main__':
    args = config.get_base_parser().parse_args()

    with train_helper.experiment(args, args.save_prefix) as e:
        np.random.seed(e.config.random_seed)
        torch.manual_seed(e.config.random_seed)

        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        e.log.info(args)
        e.log.info("*" * 25 + " ARGS " + "*" * 25)

        res = run(e)
        pickle.dump(res, open(e.experiment_dir + "/theta+weights+netD+state-dict.pkl", "wb+"))
