import torch
import config
# import data_utils
import train_helper
# import models
import pickle
import numpy as np
from train import run


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
            all_theta_list.append(theta_[0].mean(0).astype("float32"))
            e.log.info("finished {}: theta_list = {}".format(i, theta_))
        pickle.dump(all_theta_list,
                    open(e.experiment_dir + "/all_thetas.pkl", "wb+"))
