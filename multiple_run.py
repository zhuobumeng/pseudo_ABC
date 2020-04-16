import argparse
import multiple_run_scripts as scripts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_dim', type=int, default=5)
    parser.add_argument('--mean1', type=float, default=0.0)
    parser.add_argument('--variance1', type=float, default=1.0)
    parser.add_argument('--mean2', type=float, default=1.0)
    parser.add_argument('--variance2', type=float, default=1.0)
    parser.add_argument('--mix1', type=float, default=1.0)
    parser.add_argument('--mix2', type=float, default=0.0)
    parser.add_argument("--random_seed",
                        type=int, default=0, help="random seed")
    parser.add_argument("--how_many_each",
                        type=int, default=50, help="how many replicates each")
    parser.add_argument("--how_many_rounds",
                        type=int, default=10,
                        help="how many rounds of replicates")
    parser.add_argument("--penalty", default=0.01, type=float, help="penalty")
    parser.add_argument("--d_hidden_size", dest="dhs", default=50,
                        type=int, help="d hidden size")
    parser.add_argument("--n_epoch", default=2000, type=int)
    parser.add_argument("--data_size", default=10000, type=int)
    parser.add_argument("--note", default="", type=str)
    parser.add_argument('--var', type=str, nargs="+", default=["a"],
                        help='choose inside a, b, g, k')
    parser.add_argument("--init", type=float, default=[2.5], nargs="+",
                        help="init value of var")
    parser.add_argument("--truth", type=float, default=[3.0, 2.0, 1.0, 0.5],
                        nargs="+", help="true parameters")
    parser.add_argument("--model", type=str, default="gnk",
                        help="choose from gnk, gaussian")
    args = parser.parse_args()

    if args.model.lower() == "gnk":
        run = scripts.run_gnk
    if args.model.lower() == "gaussian":
        run = scripts.run_gaussian
    else:
        ValueError("Input model is not in the scripts!")

    run(vars(args))
