import numpy as np

from main import get_noisy_sample_sigma_per_layer, get_layer_dict, get_datasets, prune_function, \
    remove_reparametrization, test
import argparse

import glob
import omegaconf
from alternate_models.resnet import ResNet50_rf, ResNet18_rf, ResNet24_rf
from alternate_models.vgg import VGG_RF
import torch.nn as nn
import torch
from torchvision.models import resnet18, resnet50
import pandas as pd
import re
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):

    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": args.model,
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": args.pruning_rate,
         "noise": "gaussian",
         "sigma": args.sigma,
         "pruner": "global",
         "exclude_layers": exclude_layers

         })
    train, val, testloader = get_datasets(cfg)

    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, rf_level=args.RF_level)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg19":
        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, rf_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, rf_level=args.RF_level)
    if args.model == "resnet24":

        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))

    dense_accuracy_list = []
    pruned_accuracy_list = []
    files_names = []
    search_string = "{}/{}_normal_{}_*_level_{}*{}*test_acc*.pth".format(args.folder, args.model, args.dataset,
                                                                         args.RF_level, args.name)
    things = list(glob.glob(search_string))
    if len(things) < 2:
        search_string = "{}/{}_normal_{}_*_level_{}.pth".format(args.folder, args.model, args.dataset, args.RF_level)

    print("Glob text:{}".format(
        "{}/{}_normal_{}_*_level_{}*test_acc*.pth".format(args.folder, args.model, args.dataset, args.RF_level)))
    print(things)
    for i, name in enumerate(
            glob.glob(search_string)):
        print(name)
        if "width" in name:
            continue
        state_dict_raw = torch.load(name, map_location=device)
        dense_accuracy_list.append(state_dict_raw["acc"])
        print("Dense accuracy:{}".format(state_dict_raw["acc"]))
        net.load_state_dict(state_dict_raw["net"])

        ################## add noise to the model
        names, weights = zip(*get_layer_dict(net))
        number_of_layers = len(names)
        sigma_per_layer = dict(zip(names, [cfg.sigma] * number_of_layers))
        stochastic_pruning_accuracies = []
        for i in range(10):
            new_net = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
            prune_function(new_net, cfg)
            remove_reparametrization(new_net, exclude_layer_list=cfg.exclude_layers)
            pruned_accuracy = test(new_net, use_cuda=False, testloader=testloader, verbose=0)
            stochastic_pruning_accuracies.append(pruned_accuracy)
        median_stochastic_accuracy = np.median(stochastic_pruning_accuracies)

        print("Median stochastic accuracy:{}".format(median_stochastic_accuracy))

        pruned_accuracy_list.append(median_stochastic_accuracy)
        weight_names, weights = zip(*get_layer_dict(net))
        zero_number = lambda w: (torch.count_nonzero(w == 0) / w.nelement()).cpu().numpy()
        pruning_rates_per_layer = list(map(zero_number, weights))

        # seed_from_file1 = re.findall("_[0-9]_", name)[0].replace("_", "")
        #
        # seed_from_file2 = re.findall("_[0-9]_[0-9]_", name)
        #
        # if seed_from_file2:
        #     seed_from_file = seed_from_file2[0].split("_")[2]
        # else:
        #     seed_from_file = seed_from_file1
        #
        # df2 = pd.DataFrame({"layer_names": weight_names, "pr": pruning_rates_per_layer})
        # df2.to_csv(
        #     "{}_level_{}_seed_{}_{}_pruning_rates_global_pr_{}.csv".format(args.model, args.RF_level, seed_from_file,
        #                                                                    args.dataset, args.pruning_rate),
        #     index=False)
        print("Done")
        file_name = os.path.basename(name)
        print(file_name)
        files_names.append(file_name)

    df = pd.DataFrame({"Name": files_names,
                       "Dense Accuracy": dense_accuracy_list,
                       "Pruned Accuracy": pruned_accuracy_list,
                       })
    df.to_csv("SP_RF_{}_{}_{}_{}_one_shot_sigma_{}_summary.csv".format(args.model, args.RF_level, args.dataset, args.pruning_rate,args.sigma),
              index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='One shot pruning statistics')

    parser.add_argument('--experiment', default=2, type=int, help='Experiment to perform')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--type', default="normal", type=str, help='Type of implementation [normal,official]')
    parser.add_argument('--RF_level', default=4, type=int, help='Receptive field level')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', default="tiny_imagenet", type=str, help='Dataset to use [cifar10,tiny_imagenet616gg]')
    parser.add_argument('--model', default="resnet50", type=str, help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--folder', default="trained_models", type=str,
                        help='Location where saved models are')
    parser.add_argument('--name', default="", type=str, help='Name of the file', required=False)
    parser.add_argument('--solution', default="", type=str, help='Solution to use')
    parser.add_argument('--pruning_rate', default=0.9, type=float, help='Pruning rate')
    parser.add_argument('--epochs', default=200, type=int, help='Epochs to train')
    parser.add_argument('--sigma', default=0.005, type=float, help='Epochs to train')
    args = parser.parse_args()
    main(args)
