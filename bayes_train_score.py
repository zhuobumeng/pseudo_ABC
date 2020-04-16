import torch
import config
import train_helper
import bayes_model_score

import pickle
import numpy as np


def run(e):
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    if e.config.model.lower() == "score_gaussian":
        model_class = bayes_model_score.gaussian_dist
    if e.config.model.lower() == "score_exp":
        model_class = bayes_model_score.exp_dist
    if e.config.model.lower() == "score_bernoulli":
        model_class = bayes_model_score.bernoulli_dist
    if e.config.model.lower() == "score_poisson":
        model_class = bayes_model_score.poisson_dist

    model = model_class(
        input_dim=e.config.isize,
        experiment=e)

    theta_true = np.ones(
        shape=(1, e.config.isize)).astype("float32") * e.config.m1
    mimic_data = model.gen_data(theta_true, e.config.data_size)
    theta_y = model.estimate(mimic_data)

    e.log.info("Estimate:" + str(theta_y.flatten()))

    e.log.info(model)

    model.train()

    model.get_theta(theta_y)

    model.save()

    pickle.dump([
        theta_true, theta_y, model.all_delta,
        model.all_weights, model.all_grad_norm, model.all_theta],
        open(e.experiment_dir + "/bayes_all+thetas+weights+norm.pkl", "wb+"))


if __name__ == '__main__':
    args = config.get_base_parser().parse_args()

    with train_helper.experiment(args, args.save_prefix) as e:
        np.random.seed(e.config.random_seed)
        torch.manual_seed(e.config.random_seed)

        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        e.log.info(args)
        e.log.info("*" * 25 + " ARGS " + "*" * 25)

        run(e)
