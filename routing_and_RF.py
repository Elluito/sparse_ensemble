import os
import copy
import time
import torch
import re
import argparse
import glob
import torchvision.transforms as transforms
import torchvision
from pathlib import Path
from alternate_models import *
import pandas as pd
from main import prune_function, remove_reparametrization, get_layer_dict, get_datasets, count_parameters
import omegaconf
from similarity_comparison_architecture import features_similarity_comparison_experiments
import numpy as np
from torch.nn.utils import parameters_to_vector
from sparse_ensemble_utils import disable_bn, mask_gradient, sparsity
import random
from confidence_utils import check_none_and_replace

from thop import profile
import matplotlib
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fs=12
def plot_cascade():
    acc_max_prob = [0.73304, 0.73984, 0.74582, 0.75062, 0.75536, 0.75912, 0.76176, 0.76252, 0.76336, 0.76354, 0.76316,
                    0.76256, 0.76206, 0.76166, 0.7615, 0.76138, 0.76128, 0.7613, 0.76128, 0.76128, 0.76128]
    acc_logits_gap = [0.73304, 0.74052, 0.74638, 0.7524, 0.75732, 0.75996, 0.76124, 0.76242, 0.76316, 0.7632, 0.76304,
                      0.76252, 0.76222, 0.76178, 0.76152, 0.76138, 0.76134, 0.7613, 0.76128, 0.76128, 0.76128]
    acc_neg_entropy = [0.73304, 0.73854, 0.74318, 0.7475, 0.75222, 0.75472, 0.75806, 0.7606, 0.76172, 0.7632, 0.76318,
                       0.76264, 0.7621, 0.7616, 0.76148, 0.76134, 0.7613, 0.76128, 0.76128, 0.76128, 0.76128]

    acc_f = 73.314
    acc_s = 76.13
    macs_f = 3675629032.0
    macs_s = 4121925096.0

    x = np.arange(0, 105, 5)
    # interval = np.arange(0, 1.05, 0.05)
    # x = macs_f + interval * macs_f
    lw = 0.5
    ms = 2
    plt.plot(x, np.array(acc_max_prob) * 100, 'o--', markersize=ms, label='Max. Prob.', c=plt.cm.Set1(2), linewidth=lw)
    # plt.plot(x, np.array(acc_2__1)*100,  'o--', markersize=ms,  label='$T=0.5$', c=plt.cm.Set1(2), linewidth=lw)
    plt.plot(x, np.array(acc_logits_gap) * 100, 'o--', markersize=ms, label='Logits Gap', c=plt.cm.Set1(3),
             linewidth=lw)
    plt.plot(x, np.array(acc_neg_entropy) * 100, 'o--', markersize=ms, label='Negative Entropy', c=plt.cm.Set1(4),
             linewidth=lw)
    # / (10 ** 9)

    plt.plot(0, acc_f, marker="o", markersize=3, markeredgecolor="black",
             markerfacecolor="black")
    plt.annotate('R34', (3, acc_f))
    plt.plot(100, acc_s, marker="o", markersize=3, markeredgecolor="black",
             markerfacecolor="black")
    plt.annotate('R50', (100 - 8, acc_s + 0.1))

    plt.grid(ls='--', alpha=0.5)

    plt.xlabel('$\%$ images routed into ResNet-50', size=fs, labelpad=3)
    plt.xticks(fontsize=fs)
    plt.ylabel('Accuracy (\%)', size=fs, labelpad=3)
    plt.yticks(fontsize=fs)

    legend = plt.legend(loc='lower right', edgecolor='black', fontsize=fs - 1)
    legend.get_frame().set_linewidth(0.5)

    plt.savefig(f"./assets/resnet34-50-cas-imagenet-conf.pdf", bbox_inches='tight', pad_inches=0.05)

    plt.show()
def check_correctness(outputs, targets):
    correct_soft_max = 0
    soft_max_outputs = F.softmax(outputs, dim=1)
    print("soft_max:{}".format(soft_max_outputs))
    _, predicted = torch.max(outputs.data, 1)
    soft_max_pred, predicted_soft_max = torch.max(soft_max_outputs.data, 1)
    # total += targets.size(0)
    correct_list = predicted.eq(targets.data).cpu()

    correct_soft_max += predicted_soft_max.eq(targets.data).cpu().sum()

    return correct_list, soft_max_pred


def cascade_accuracy(max_prob1, correct1, correct2, macs_m1, macs_m2):
    accuracy = []
    total_macs = []
    index = np.argsort(max_prob1)
    proportion = np.linspace(0, 1, num=21)
    correct1 = correct1[index]
    correct2 = correct2[index]

    for p in proportion:
        cascade_correct = correct1.copy()

        # accuracy for cascade

        cascade_correct[:int(p * len(correct1))] = correct2[:int(p * len(correct1))]

        accuracy.append(np.mean(cascade_correct))
        total_macs.append(p * len(correct1) * macs_m1 + (1 - p) * len(correct2) * macs_m2)

        # print('p:', p, 'acc:', accuracy)
    return accuracy, total_macs


def swapping(args):
    def generate_binary_list(length, rate):
        # Initialize a list of ones
        binary_list = [1] * length
        num_zeros = int(rate * length)

        # Set a specified number of elements to 0 in random positions
        zeros_indices = random.sample(range(length), num_zeros)
        for idx in zeros_indices:
            binary_list[idx] = 0

    theta_1 = torch.load(os.path.join(args.model_location, f"{args.sd1}.pt"),
                         map_location=torch.device('cpu'))

    theta_2 = torch.load(os.path.join(args.model_location, f"{args.sd2}.pt"),
                         map_location=torch.device('cpu'))
    swapping_rate = 0.5

    acc_theta_1 = []
    acc_theta_2 = []
    # random recombination
    num_elements = sum(p.numel() for p in theta_1.values())
    swap_list = generate_binary_list(num_elements, rate=swapping_rate)

    theta_1_swap, theta_2_swap, start = {}, {}, 0
    for k in theta_1:
        end = start + theta_1[k].numel()
        m = swap_list[start:end].reshape(theta_1[k].shape)
        # Reshape the swap mask back to the original shape of the parameters
        theta_1_swap[k] = theta_1[k] * m + theta_2[k] * (1 - m)
        theta_2_swap[k] = theta_2[k] * m + theta_1[k] * (1 - m)
        start = end


def load_model(args):
    from torchvision.models import resnet18, resnet50
    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
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
            net = VGG_RF("VGG19_rf", num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
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
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type))
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
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "deep_resnet_small":
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
    if args.model == "vgg19_stride":
        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF_stride("VGG19_rf", num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF_stride("VGG19_rf", num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF_stride("VGG19_rf", num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = VGG_RF_stride("VGG19_rf", num_classes=200, RF_level=args.RF_level)

    if args.solution:
        temp_dict = torch.load(args.solution, map_location=torch.device('cpu'))["net"]
        if args.modeltype1 == "normal" and args.rf_level != 0:
            net.load_state_dict(temp_dict)
            print("loaded solution!")
        else:
            real_dict = {}
            for k, item in temp_dict.items():
                if k.startswith('module'):
                    new_key = k.replace("module.", "")
                    real_dict[new_key] = item
            net.load_state_dict(real_dict)
            print("loaded solution!")
    return net


def check_correctness_dataloaders(model1, model2, dataloader, device, topk=5):
    model1 = model1.to(device)
    model2 = model2.to(device)
    model1.eval()
    model2.eval()

    full_accuracies = None
    full_confidences = None
    full_max_prob_model1 = []
    full_correct_model1 = []
    full_correct_model2 = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        outputs1 = model1(x)
        outputs2 = model2(x)
        correct1, confidences1 = check_correctness(outputs1, y)
        correct2, confidences2 = check_correctness(outputs2, y)
        full_max_prob_model1.extend(confidences1)
        full_correct_model1.extend(correct1)
        full_correct_model2.extend(correct2)
    return full_max_prob_model1, full_correct_model1, full_correct_model2


def main(args):
    if "vgg" in args.model:
        exclude_layers = ["features.0", "classifier"]
    if "resnet" in args.model:
        exclude_layers = ["conv1", "linear"]
    if "densenet" in args.model:
        exclude_layers = ["conv1", "fc"]
    if "resnet" in args.model:
        exclude_layers = ["conv1", "linear"]
    if "mobilenet" in args.model:
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
         "sigma": 0.005,
         # "pruner": "global",
         "pruner": "lamp",
         "exclude_layers": exclude_layers,
         "data_path": args.data_folder,
         "input_resolution": args.input_resolution

         })

    if args.ffcv:
        from ffcv_loaders import make_ffcv_small_imagenet_dataloaders
        train, val, testloader = make_ffcv_small_imagenet_dataloaders(args.ffcv_train, args.ffcv_val,
                                                                      128, args.num_workers)
    else:
        print("Normal data loaders loaded!!!!")
        cifar10_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        cifar100_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        stats_to_use = cifar10_stats if args.dataset == "cifar10" else cifar100_stats
        # Data
        print('==> Preparing data..')
        current_directory = Path().cwd()
        data_path = "."
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
            data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
        elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
            data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "/home/luisaam/Documents/PhD/data/"
        print(data_path)
        batch_size = args.batch_size
        if "32" in args.name:
            batch_size = 32
        if "64" in args.name:
            batch_size = 64

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
                trainloader, valloader, testloader = make_ffcv_small_imagenet_dataloaders(args.ffcv_train,
                                                                                          args.ffcv_val,
                                                                                          batch_size, args.num_workers)
            else:
                from test_imagenet import load_small_imagenet
                trainloader, valloader, testloader = load_small_imagenet(
                    {"traindir": data_path + "/small_imagenet/train", "valdir": data_path + "/small_imagenet/val",
                     "num_workers": args.num_workers, "batch_size": batch_size, "resolution": args.input_resolution})

    cfg_1 = omegaconf.DictConfig(
        {"architecture": args.model,
         "type": "normal",
         # "model_type": "hub",
         "RF_level": args.RF_level1,
         "solution": args.solution1,
         # "solution": "trained_m
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": args.pruning_rate,
         "noise": "gaussian",
         "sigma": 0.005,
         # "pruner": "global",
         "pruner": "lamp",
         "exclude_layers": exclude_layers,
         "data_path": args.data_folder,
         "input_resolution": args.input_resolution

         })

    model1 = load_model(cfg_1)

    cfg_2 = omegaconf.DictConfig(
        {"architecture": args.model,
         "type": "normal",
         "RF_level": args.RF_level2,
         "solution": args.solution2,
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": args.pruning_rate,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "lamp",
         "exclude_layers": exclude_layers,
         "data_path": args.data_folder,
         "input_resolution": args.input_resolution
         })

    model2 = load_model(cfg_2)

    x, y = next(iter(trainloader))

    x = x.to(device)

    input = torch.randn(1, 3, 224, 224)

    input = input.to(device)

    macs_one_image_m1, params = profile(copy.deepcopy(model1), inputs=(input,))
    macs_one_image_m2, params = profile(copy.deepcopy(model2), inputs=(input,))

    max_prob1, correct1, correct2 = check_correctness_dataloaders(model1, model2, testloader, device)

    accuracy_cascade,macs_cascade = cascade_accuracy(max_prob1, correct1, correct2,macs_one_image_m1,macs_one_image_m2)


    # dense_accuracy_list = []
    #
    # pruned_accuracy_list = []
    #
    # files_names = []
    #
    # search_string = "{}/{}_normal_{}_*_level_{}*{}*test_acc_*.pth".format(args.folder, args.model, args.dataset,
    #                                                                       args.RF_level, args.name)
    #
    # things = list(glob.glob(search_string))
    #
    # # if len(things) < 2:
    # #     search_string = "{}/{}_normal_{}_*_level_{}.pth".format(args.folder, args.model, args.dataset, args.RF_level)
    #
    # print("Glob text:{}".format(
    #     "{}/{}_normal_{}_*_level_{}*{}*test_acc_*.pth".format(args.folder, args.model, args.dataset, args.RF_level,
    #                                                           args.name)))
    # print(things)
    #
    # for i, name in enumerate(
    #         glob.glob(search_string)):
    #     print(name)
    #
    #     print("Device: {}".format(device))
    #
    #     state_dict_raw = torch.load(name, map_location=device)
    #
    #     net.load_state_dict(state_dict_raw["net"])
    #
    #     print("Dense accuracy:{}".format(state_dict_raw["acc"]))
    #
    #     calculated_accuracy = test(net, testloader=testloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Routing and RF ')
    parser.add_argument('--experiment', default=1, type=int, help='Experiment to perform')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--type', default="normal", type=str, help='Type of implementation [normal,official]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('-s1', '--solution1', type=str, default="", help='',
                        required=False)
    parser.add_argument('-s2', '--solution2', type=str, default="", help='',
                        required=False)
    parser.add_argument('--RF_level1', default="4", type=str, help='Receptive field level')
    parser.add_argument('--RF_level2', default="4", type=str, help='Receptive field level')

    parser.add_argument('--dataset', default="cifar10", type=str, help='Dataset to use [cifar10,tiny_imagenet616gg]')
    parser.add_argument('--model', default="resnet18", type=str, help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where saved models are')
    parser.add_argument('--save_folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Output folder of the pruning results')
    parser.add_argument('--data_folder', default="/nobackup/sclaam/data", type=str,
                        help='Location to save the models', required=True)
    parser.add_argument('--name', default="", type=str, help='Name of the file', required=False)
    parser.add_argument('--solution', default="", type=str, help='Solution to use')
    parser.add_argument('--pruning_rate', default=0.9, type=float, help='Pruning rate')
    parser.add_argument('--epochs', default=200, type=int, help='Epochs to train')
    parser.add_argument('--width', default=1, type=int, help='Width of the model')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size for training')
    parser.add_argument('--input_resolution', default=224, type=int,
                        help='Input Resolution for small ImageNet')
    parser.add_argument('--ffcv', action='store_true', help='Use FFCV loaders')

    parser.add_argument('--ffcv_train',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv",
                        type=str, help='FFCV train dataset')

    parser.add_argument('--ffcv_val',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv",
                        type=str, help='FFCV val dataset')

    args = parser.parse_args()
