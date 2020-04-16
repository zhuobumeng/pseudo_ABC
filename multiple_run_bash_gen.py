import numpy as np
import argparse
import os


def run(config):
    print(config)
    params_init = {
        "a": "1 2 1 0.5",
        "b": "3 1 1 0.5",
        "g": "3 2 0.15 0.5",
        "k": "3 2 1 0.1"
    }
    for i in range(config["how_many_rounds"]):
        random_seed = config["random_seed"] + i
        config["job_name"] = "gnkmul" + str(random_seed)
        config["job_id"] = "gnkmul" + str(random_seed)
        file_name = "run-gnk-multiple-py" + str(random_seed) + ".sh"
        with open(file_name, "w") as file:
            file.write(f'''#!/bin/bash

#SBATCH --job-name={config["job_name"]}
#SBATCH --output={config["job_id"]}.out
#SBATCH --error={config["job_id"]}.err
#SBATCH --nodes=1
#SBATCH --partition=broadwl
#SBATCH --ntasks=1

module load cuda/10.0

python train_gnk_multiple.py \\
--debug 0 \\
--save_prefix gnk_multiple_feb5/gnk_{config["var"]} \\
--model tv_pen \\
--d_hidden_size 20 \\
--data_size 10000 \\
-params 3 2 1 0.5 \\
-params_init {params_init[config["var"]]} \\
--n_epoch 1000 \\
--summarize 0 \\
--opt adam \\
--learning_rate_dis 1e-2 \\
--learning_rate_gen 1e-2 \\
--d_step 15 \\
--batch_size 100 \\
--sample_size 10000 \\
--latent_z_size 1 \\
--input_size 1 \\
--penalty 0.01 \\
--activate tanh \\
--num_run {config["how_many_each"]} \\
--random_seed {config["random_seed"]}
''')
        os.system("chmod +x " + file_name)
        print("sbatch " + file_name)
        os.system("sbatch " + file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--var', type=str, default=None,
                        help='choose inside a, b, g, k')
    parser.add_argument("--random_seed",
                        type=int, default=0, help="random seed")
    parser.add_argument("--how_many_each",
                        type=int, default=50, help="how many replicates each")
    parser.add_argument("--how_many_rounds",
                        type=int, default=10,
                        help="how many rounds of replicates")
    args = parser.parse_args()

    run(vars(args))
