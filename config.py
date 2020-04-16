import argparse


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    basic_group = parser.add_argument_group('basics')
    # Basics
    basic_group.add_argument('--debug', type="bool", default=False,
                             help='activation of debug mode (default: False)')
    basic_group.add_argument('--save_prefix', type=str, default="experiments",
                             help='saving path prefix')
    basic_group.add_argument('--is_bayes', type="bool", default=False,
                             help='activation of bayesian training')

    data_group = parser.add_argument_group('data')
    # Data file
    data_group.add_argument("-params", "--parameters", dest="params",
                            type=float, nargs="+", default=[1.0])
    data_group.add_argument("-params_init", "--parameters_init",
                            dest="params_init",
                            type=float, nargs="+", default=[1.0])
    data_group.add_argument('--train_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--weight_path', type=str, default=None,
                            help='weight file')
    config_group = parser.add_argument_group('model_configs')
    config_group.add_argument('--random_seed', type=int, default=0,
                              help='Random seed')
    config_group.add_argument('--data_size', type=int, default=10000,
                              help='data size', dest='data_size')
    config_group.add_argument('--model',
                              type=str, default="tv",
                              # choices=['tv', 'js', 'freq', 'freqb',
                              #          'freqc', 'freqd', 'freqe',
                              #          'freqf', 'freqg', 'bayes1',
                              #          'bayes2', 'bayes3'],
                              help='types of model')
    config_group.add_argument('-isize', '--input_size',
                              dest='isize',
                              type=int,
                              default=100)
    config_group.add_argument('-ssize', '--sample_size',
                              dest='ssize',
                              type=int,
                              default=100)
    config_group.add_argument('-osize', '--output_size',
                              dest='osize',
                              type=int,
                              default=1)
    config_group.add_argument('-bsize', '--batch_size',
                              dest='bsize',
                              type=int,
                              default=100)
    config_group.add_argument('-dhsize', '--d_hidden_size',
                              dest='dhsize',
                              type=int,
                              default=0)
    config_group.add_argument('-ghsize', '--g_hidden_size',
                              dest='ghsize',
                              type=int,
                              default=0)
    config_group.add_argument('-zsize', '--latent_z_size',
                              dest='zsize',
                              type=int,
                              default=50)
    config_group.add_argument('-ds', '--d_step',
                              dest='ds',
                              type=int,
                              default=5)
    config_group.add_argument('-pt', '--penalty',
                              dest='pt',
                              type=float,
                              default=0.1)
    config_group.add_argument('-pt_all', '--penalty_on_all',
                              dest="pt_all", type="bool", default=True)
    config_group.add_argument('-nt', '--n_theta',
                              dest='nt',
                              type=int,
                              default=500)
    config_group.add_argument('--eps',
                              type=float,
                              default=1e-6)
    config_group.add_argument('-std', '--sample_standard_deviation',
                              dest='std',
                              type=float,
                              default=0.1)
    # config_group.add_argument('-a', '--a',
    #                           dest='a',
    #                           type=float,
    #                           default=3.)
    # config_group.add_argument('-b', '--b',
    #                           dest='b',
    #                           type=float,
    #                           default=1.)
    # config_group.add_argument('-g', '--g',
    #                           dest='g',
    #                           type=float,
    #                           default=2.)
    # config_group.add_argument('-k', '--k',
    #                           dest='k',
    #                           type=float,
    #                           default=0.5)
    config_group.add_argument('-m1', '--mean1',
                              dest='m1',
                              type=float,
                              default=0.0)
    config_group.add_argument('-v1', '--variance1',
                              dest='v1',
                              type=float,
                              default=1.0)
    config_group.add_argument('-m2', '--mean2',
                              dest='m2',
                              type=float,
                              default=1.0)
    config_group.add_argument('-v2', '--variance2',
                              dest='v2',
                              type=float,
                              default=1.0)
    config_group.add_argument('-mi1', '--mix1',
                              dest='mi1',
                              type=float,
                              default=1.0)
    config_group.add_argument('-mi2', '--mix2',
                              dest='mi2',
                              type=float,
                              default=0.0)
    # config_group.add_argument('-wt', '--weight_threshold',
    #                           dest='wt',
    #                           type=float,
    #                           default=1)
    # config_group.add_argument('--err', type=float,
    #                           default=0.1)
    # config_group.add_argument('-t', '--threshold',
    #                           dest='t',
    #                           type=float,
    #                           default=0.01)
    config_group.add_argument('--n_epoch', type=int, default=20,
                              help='number of epochs')
    config_group.add_argument('--note', type=str, default="",
                              help='notes you want to add?')
    config_group.add_argument("-act", "--activate", dest='activate', type=str,
                              default='Tanh')

    misc_group = parser.add_argument_group('misc')
    # misc
    misc_group.add_argument('--total_sample', type=int, default=10000,
                            help='total number of samples')
    misc_group.add_argument('--eff_prop', type=float, default=0.5,
                            help='proportion of effective sample size')
    misc_group.add_argument('--smc_T', type=int, default=20,
                            help='total round of SMC')
    misc_group.add_argument('--smc_eps_ratio', type=float, default=0.9,
                            help='smc eps ratio')
    misc_group.add_argument('--n_rounds', type=int, default=5,
                            help='number of rounds')
    misc_group.add_argument('--nit_per_round', type=int, default=1000,
                            dest="n_prnd",
                            help='number of iterations per round')
    misc_group.add_argument('-lr_s', '--learning_rate_s',
                            dest='lr_s',
                            type=float,
                            default=1.0)
    misc_group.add_argument('-nt_pertheta', '--num_per_theta',
                            dest='nt_pertheta',
                            type=int,
                            default=1)
    misc_group.add_argument('-lr_g', '--learning_rate_gen',
                            dest='lr_g',
                            type=float,
                            default=0.02)
    misc_group.add_argument('-lr_d', '--learning_rate_dis',
                            dest='lr_d',
                            type=float,
                            default=0.2)
    misc_group.add_argument('--n_iteration', type=int, default=10000,
                            help='number of iterations')
    misc_group.add_argument('--patience', type=int, default=20,
                            help='patience of early stop.')
    misc_group.add_argument('--opt', type=str, default='adam',
                            choices=['adam', 'sgd', 'rmsprop'],
                            help='types of optimizer: adam (default), \
                            sgd, rmsprop')
    misc_group.add_argument('--pretrain_it', type=int, default=500,
                            help='number of pretraining iteration')
    misc_group.add_argument('--print_every', type=int, default=100,
                            help='print training details after \
                            this number of iterations')
    misc_group.add_argument('--eval_every', type=int, default=500,
                            help='evaluate model after \
                            this number of iterations')
    misc_group.add_argument('--summarize', type="bool", default=False,
                            help='whether to summarize training stats\
                            (default: False)')
    misc_group.add_argument("-num_run", "--num_run", dest="num_run",
                            default=1, type=int,
                            help="how many times to run the algorithm.")
    return parser
