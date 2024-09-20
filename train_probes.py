# Taken from
import os
import argparse
from multiprocessing import cpu_count
import itertools
from phd_lab.experiments.probe_training import main, PseudoArgs, parse_model
from phd_lab.experiments.utils import config as cfg


def test_local():
    pargs = PseudoArgs(model_name="resnet50",
                       folder=os.path.join("./latent_datasets", f'resnet50_cifar10_32_2'),
                       mp=0)

    print(pargs)
    main(pargs)


def run_training(args):
    pargs = PseudoArgs(model_name=args.model,
                       folder=os.path.join("./latent_datasets",
                                           f'{args.model}_{args.dataset}_{args.input_resolution}_{args.RF_level}'),
                       mp=args.mp, save_path=args.save_paths)

    print(pargs)

    main(pargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', dest='model', type=str, default="vgg19", help='Model architecture')
    parser.add_argument('-d', dest='dataset', type=str, default="cifar10", help='Dataset to use')
    parser.add_argument('--RF_level', dest='RF_level', type=int, default=2, help='Receptive field level')
    parser.add_argument('--input_resolution', dest='input_resolution', type=int, default=32, help='Input resolution')
    parser.add_argument('-mp', dest='mp', type=int, default=cpu_count(), help='Enable multiprocessing')
    parser.add_argument('--save_path', default="./probes_logs/", type=str, help='Save path of logs')


    parser.add_argument('--prefix', dest='prefix', type=str, default=None, help='Postfix added to the result csv')
    parser.add_argument('-f', dest='folder', type=str, default="./latent_datasets", help='data folder')
    parser.add_argument('--config', dest='config', type=str, default=None, help='Path to a config file')
    parser.add_argument('--verbose', dest='verbose', type=bool, default=False, help='Show Epoch counter')
    parser.add_argument('--overwrite', dest='overwrite', type=bool, default=False, help='Overwrite existing results')

    args = parser.parse_args()
    # args = parser.parse_args()
    if args.prefix is not None:
        cfg.PROBE_PERFORMANCE_SAVEFILE = args.prefix + "_" + cfg.PROBE_PERFORMANCE_SAVEFILE
    if args.config is not None:
        import json
        config = json.load(open(args.config, 'r'))
        for (model, dataset, resolution) in itertools.product(config['model'], config['dataset'], config["resolution"]):
            model_name = parse_model(model, (32, 32, 3), 10)

            pargs = PseudoArgs(model_name=model_name,
                       folder=os.path.join(args.folder, f'{model_name}_{dataset}_{resolution}'),
                       mp=args.mp)

            print(pargs)
            main(pargs)
    else:
        # pargs = PseudoArgs(model_name=args.model,
        #                    folder=args.save_path,
        #                    mp=args.mp)
        main(args)
    # test_local()
