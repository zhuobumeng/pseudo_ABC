import os


def run_gnk(config):
    truth_str = " ".join(map(str, config["truth"]))
    error_dict = dict(zip(["a", "b", "g", "k"], config["truth"]))
    for x in range(len(config["var"])):
        error_dict[config["var"][x]] = config["init"][x]
    init_str = " ".join(str(error_dict[x]) for x in ["a", "b", "g", "k"])
    common_name = "gnkmul-pen" + str(config["penalty"]) \
        + "-dhs" + str(config["dhs"])
    var_str = "".join(config["var"])
    for i in range(config["how_many_rounds"]):
        random_seed = config["random_seed"] + i
        config["job_name"] = "gnkmul" + str(random_seed)
        config["job_id"] = "gnkmul" + str(random_seed)
        file_name = common_name + str(random_seed) + ".sh"
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
--save_prefix {common_name}/gnk{var_str}_{config["note"]} \\
--model tv_pen \\
--d_hidden_size {config["dhs"]} \\
--data_size {config["data_size"]} \\
-params {truth_str} \\
-params_init {init_str} \\
--n_epoch {config["n_epoch"]} \\
--summarize 0 \\
--opt adam \\
--learning_rate_dis 1e-2 \\
--learning_rate_gen 1e-2 \\
--d_step 15 \\
--batch_size 100 \\
--sample_size 10000 \\
--latent_z_size 1 \\
--input_size 1 \\
--penalty {config["penalty"]} \\
--activate tanh \\
--num_run {config["how_many_each"]} \\
--random_seed {random_seed} \\
--penalty_on_all True
''')
        os.system("chmod +x " + file_name)
        print("sbatch " + file_name)
        os.system("sbatch " + file_name)


def run_gaussian(config):
    common_name = "tvmul-pen" + str(config["penalty"]) \
        + "-dhs" + str(config["dhs"])
    for i in range(config["how_many_rounds"]):
        random_seed = config["random_seed"] + i
        config["job_name"] = "tvpenmul" + str(random_seed)
        config["job_id"] = "tvpenmul" + str(random_seed)
        file_name = common_name + str(random_seed) + ".sh"
        with open(file_name, "w") as file:
            file.write(f'''#!/bin/bash

#SBATCH --job-name={config["job_name"]}
#SBATCH --output={config["job_id"]}.out
#SBATCH --error={config["job_id"]}.err
#SBATCH --nodes=1
#SBATCH --partition=broadwl
#SBATCH --ntasks=1

module load cuda/10.0

python train_gaussian_multiple.py \\
--debug 0 \\
--save_prefix {common_name}/{config["note"]} \\
--model tv_pen \\
--mean1 {config["mean1"]} \\
--mean2 {config["mean2"]} \\
--variance1 {config["variance1"]} \\
--variance2 {config["variance2"]} \\
--mix1 {config["mix1"]} \\
--mix2 {config["mix2"]} \\
--input_size {config["param_dim"]} \\
--latent_z_size {config["param_dim"]} \\
--d_hidden_size {config["dhs"]} \\
--data_size {config["data_size"]} \\
--n_epoch {config["n_epoch"]} \\
--summarize 0 \\
--opt adam \\
--learning_rate_dis 1e-2 \\
--learning_rate_gen 1e-2 \\
--d_step 15 \\
--batch_size 100 \\
--sample_size 10000 \\
--penalty {config["penalty"]} \\
--activate tanh \\
--num_run {config["how_many_each"]} \\
--random_seed {random_seed} \\
--penalty_on_all True
''')
        os.system("chmod +x " + file_name)
        print("sbatch " + file_name)
        os.system("sbatch " + file_name)
