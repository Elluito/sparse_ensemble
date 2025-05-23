from zero_cost_nas.foresight.pruners.predictive import find_measures
import seaborn as sns
import omegaconf
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
import time
import torch.nn.init as init
import os
import argparse
from pathlib import Path
import pandas as pd
from shrinkbench.metrics.flops import flops
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from alternate_models import *
from torchconvquality import measure_quality


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))
    cifar10_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    cifar100_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    stats_to_use = cifar10_stats if args.dataset == "cifar10" else cifar100_stats

    # Data
    print('==> Preparing data..')
    # current_directory = Path().cwd()
    # data_path = "."
    # if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
    #     data_path = "/nobackup/sclaam/data"
    # elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
    #     data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
    # elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
    #     data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
    # elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
    #     data_path = "/home/luisaam/Documents/PhD/data/"
    data_path = args.data_path
    print(data_path)
    batch_size = args.batch_size

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats_to_use),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    if args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    if args.dataset == "tiny_imagenet":
        from test_imagenet import load_tiny_imagenet
        trainloader, valloader, testloader = load_tiny_imagenet(
            {"traindir": data_path + "/tiny_imagenet_200/train", "valdir": data_path + "/tiny_imagenet_200/val",
             "num_workers": args.num_workers, "batch_size": batch_size})
    if args.dataset == "small_imagenet":
        if args.ffcv:
            from ffcv_loaders import make_ffcv_small_imagenet_dataloaders
            trainloader, valloader, testloader = make_ffcv_small_imagenet_dataloaders(args.ffcv_train, args.ffcv_val,
                                                                                      batch_size, args.num_workers)
        else:
            from test_imagenet import load_small_imagenet
            trainloader, valloader, testloader = load_small_imagenet(
                {"traindir": data_path + "/small_imagenet/train", "valdir": data_path + "/small_imagenet/val",
                 "num_workers": args.num_workers, "batch_size": batch_size, "resolution": args.input_resolution})

    from torchvision.models import resnet50

    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "resnet50_stride":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf_stride(num_classes=10, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf_stride(num_classes=100, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf_stride(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet50_rf_stride(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "resnet24":

        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_eatures
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
        if args.type == "pytorch" and args.dataset == "cifar100":
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 100)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
    if args.model == "vgg19":

        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
    if args.model == "vgg19_stride":
        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF_stride("VGG19_rf", num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF_stride("VGG19_rf", num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF_stride("VGG19_rf", num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = VGG_RF_stride("VGG19_rf", num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet_small":
        if args.type == "normal" and args.dataset == "cifar10":
            net = small_ResNet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)

        if args.type == "normal" and args.dataset == "cifar100":
            net = small_ResNet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            raise NotImplementedError
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            raise NotImplementedError
    if args.model == "resnet40_small":
        if args.type == "normal" and args.dataset == "cifar10":
            net = deep_small_ResNet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = deep_small_ResNet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = deep_small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = deep_small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            raise NotImplementedError
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 100)
            raise NotImplementedError
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 100)
    if args.model == "resnet25_small":
        if args.type == "normal" and args.dataset == "cifar10":
            net = deep_2_small_Resnet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width,
                                         number_layers=25)
        if args.type == "normal" and args.dataset == "cifar100":
            net = deep_2_small_Resnet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width,
                                         number_layers=25)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = deep_2_small_Resnet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width,
                                         number_layers=25)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = deep_2_small_Resnet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width,
                                         number_layers=25)
        if args.type == "pytorch" and args.dataset == "cifar10":
            raise NotImplementedError
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 100)
            raise NotImplementedError
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg_small":

        if args.type == "normal" and args.dataset == "cifar10":
            net = small_VGG_RF("small_vgg", num_classes=10, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "cifar100":
            net = small_VGG_RF("small_vgg", num_classes=100, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_VGG_RF("small_vgg", num_classes=200, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "small_imagenet":
            net = small_VGG_RF("small_vgg", num_classes=200, RF_level=args.RF_level)
    if args.model == "densenet40":
        if args.type == "normal" and args.dataset == "cifar10":
            net = densenet_40_RF([0] * 100, num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = densenet_40_RF([0] * 100, num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = densenet_40_RF([0] * 100, num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = densenet_40_RF([0] * 100, num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "imagenet":
            net = densenet_40_RF([0] * 100, num_classes=1000, RF_level=args.RF_level)
    if args.model == "mobilenetv2":
        if args.type == "normal" and args.dataset == "cifar10":
            net = MobileNetV2_cifar_RF(num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = MobileNetV2_cifar_RF(num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = MobileNetV2_cifar_RF(num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = MobileNetV2_imagenet_RF(num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "imagenet":
            net = MobileNetV2_imagenet_RF(num_classes=1000, RF_level=args.RF_level)
    if args.model == "densenet28":
        if args.type == "normal" and args.dataset == "cifar10":
            net = densenet_28_RF([0] * 100, num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = densenet_28_RF([0] * 100, num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = densenet_28_RF([0] * 100, num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = densenet_28_RF([0] * 100, num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "imagenet":
            net = densenet_28_RF([0] * 100, num_classes=1000, RF_level=args.RF_level)

    print("Device: {}".format(device))

    dict_of_dicts = measure_quality(net.cpu())
    quality_df = pd.DataFrame(dict_of_dicts)
    quality_df = quality_df.T
    quality_df = quality_df.reset_index()
    quality_df = quality_df.rename(columns={"index": "layer_name"})
    solution_name = ["test"] * len(quality_df)
    quality_df["solution_name"] = solution_name
    # jacob_measure,snip,synflow = find_measures(net, trainloader, ("random", 16, 200), device, measure_names=["jacob_cov","snip","synflow"])
    measures = find_measures(net, trainloader, ("random", 1, 200), device,
                             measure_names=["jacob_cov", "snip", "synflow", "grad_norm", "fisher"])
    # snip = find_measures(net, trainloader, ("grasp", 10, 200), device, measure_names="snip")
    # synflow = find_measures(net, trainloader, ("grasp", 10, 200), device, measure_names="synflow")

    # return jacob_measure, snip, synflow
    # return snip, synflow
    return measures


def run_local_test():
    number_of_samples = 5
    rf_levels = [3, 4, 5, 6, 7, 8, 9, 10]
    real_ranks = [3, 2, 1, 4, 5, 6, 7, 8]
    jacob_measures = []
    snip_measures = []
    synflow_measures = []

    all_df = None
    for sample_index in range(number_of_samples):
        measures_dict = defaultdict(list)
        temp_measures = None
        for level in rf_levels:
            cfg = omegaconf.DictConfig({
                "model": "resnet_small",
                "dataset": "small_imagenet",
                "type": "normal",
                "RF_level": level,
                "lr": 0.1,
                "grad_clip": 1,
                "momentum": 0.9,
                "num_workers": 0,
                "optimiser": "sam",
                "record": False,
                "record_flops": True,
                "record_time": False,
                "use_scheduler_batch": False,
                "use_scheduler": True,
                "batch_size": 320,
                "epochs": 1,
                "width": 1,
                "input_resolution": 224,
                "name": "no_name",
                "ffcv": False,
                "save": False,
                "save_folder": "./second_order_results",
                "record_saturation": True,
                "data_path": "/home/luisaam/Documents/PhD/data/",
                # "data_path": "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets",
            })

            # jacob, snip, synflow = main(cfg)
            measures = main(cfg)
            for k, v in measures.items():
                if k == "jacob_cov":
                    measures_dict[k].extend([abs(complex(measures["jacob_cov"]))])
                else:
                    measures_dict[k].extend([measures[k]])
            # jacob_measures.append(abs(complex(measures["jacob_cov"])))
            #
            # snip_measures.append(measures["snip"])
            # #
            # synflow_measures.append(measures["synflow"])

        df = pd.DataFrame({
            "Ranks": real_ranks,
            "Rf_level": rf_levels,
            # "jacob_cov": jacob_measures,
            # "jacob_cov_ranks": np.argsort(jacob_measures)[::-1],
            # "snip": snip_measures,
            # "snip_ranks": np.argsort(snip_measures)[::-1],
            # "synflow": synflow_measures,
            # "synflow_ranks": np.argsort(synflow_measures)[::-1],
            "sample": [sample_index] * len(rf_levels),
        })
        print("df")
        print(df)
        print(v)
        for k, v in measures_dict.items():
            df["{}".format(k)] = v
            df["{}_rank".format(k)] = pd.DataFrame.rank(df["{}".format(k)], ascending=False)
        # df["jacob_cov_ranks"] = df["jacob_cov"].rank(ascending=False)
        # df["synflow_ranks"] = df["synflow_ranks"].rank(ascending=False)
        # df["snip_ranks"] = df["snip"].rank(ascending=False)
        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat((all_df, df), ignore_index=True)

    all_df.to_csv("predicting_optimal_RF/small_resnet_small_imagenet_predict_rf_more_measures_5_samples.csv", sep=",",
                  index=False)


def run_analysis_of_measures():
    from scipy.stats import spearmanr
    df_primal = pd.read_csv(
        "/home/luisaam/PycharmProjects/sparse_ensemble/predicting_optimal_RF/small_resnet_small_imagenet_predict_rf_more_measures_5_samples.csv",
        sep=",")
    measure_names = ["jacob_cov", "snip", "synflow", "grad_norm", "fisher"]
    correlation_to_actual_rank_whole = {}
    correlation_to_actual_rank_samples = defaultdict(list)
    for m in measure_names:
        df = df_primal[df_primal["Rf_level"] >= 7]
        object = spearmanr(df["Ranks"], df[m])
        statistic, p_value = object.statistic, object.pvalue

        correlation_to_actual_rank_whole[m] = statistic
        for i in range(5):
            current_df = df_primal[df_primal["sample"] == i]
            object = spearmanr(current_df["Ranks"], current_df[m])
            statistic, p_value = object.statistic, object.pvalue
            correlation_to_actual_rank_samples[m].append(statistic)

    df_whole = pd.DataFrame(
        {"Method": list(correlation_to_actual_rank_whole.keys()),
         "spearman rank correlation": list(correlation_to_actual_rank_whole.values())
         }
    )

    # df_whole.plot()
    sns.barplot(data=df_whole, x="Method", y="spearman rank correlation")
    plt.show()
    # samples_dict = {}

    # for m in measure_names:
    #
    #     samples_dict["method"]=
    #
    # df_samples = pd.DataFrame(
    #     {
    #
    #     }
    # )


if __name__ == '__main__':
    run_local_test()
    # run_analysis_of_measures()
