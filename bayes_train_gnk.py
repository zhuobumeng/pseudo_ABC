import torch
import config
import train_helper
import bayes_models_gnk as bayes_models
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

    [thetalist, netD_state, params, indices] = pickle.load(
        open(e.config.weight_path + "/theta+netD+state-dict.pkl", "rb"))

    theta_y = np.array(thetalist).mean(0).astype("float32")
    # gen_init = theta_y.copy()
    # theta_y = np.array([[0] * e.config.isize]).astype("float32")
    # gen_init = theta_y.copy()
    # gen_init = np.array([[0] * e.config.isize]).astype("float32")

    model = model_class(
        gen_init=theta_y,
        indices=indices,
        disc_init=netD_state,
        experiment=e)

    e.log.info("theta_y" + str(theta_y))
    e.log.info(model)

    model.train()
    model.get_theta()
    model.save()

    pickle.dump(
        [params, theta_y, model.all_delta, model.all_weights,
         model.all_theta, model.all_grad_norm, model.all_loss],
        open(e.experiment_dir + "/bayes_all+thetas+weights.pkl", "wb+"))


if __name__ == '__main__':

    args = config.get_base_parser().parse_args()

    with train_helper.experiment(args, args.save_prefix) as e:
        np.random.seed(e.config.random_seed)
        torch.manual_seed(e.config.random_seed)

        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        e.log.info(args)
        e.log.info("*" * 25 + " ARGS " + "*" * 25)

        run(e)
