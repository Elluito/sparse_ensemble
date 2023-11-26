import os

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
from sparse_ensemble_utils import test
import omegaconf
from similarity_comparison_architecture import features_similarity_comparison_experiments
import numpy as np
from torch.nn.utils import parameters_to_vector

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Level 0
rf_level0_s1 = "trained_models/cifar10/resnet50_cifar10.pth"
name_rf_level0_s1 = "_seed_1_rf_level_0"

rf_level0_s2 = "trained_models/cifar10/resnet50_normal_seed_2_tst_acc_95.65.pth"
name_rf_level0_s2 = "_seed_2_rf_level_0"

# level 1
rf_level1_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_1_95.26.pth"
name_rf_level1_s1 = "_seed_1_rf_level_1"
rf_level1_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_1_94.93.pth"
name_rf_level1_s2 = "_seed_2_rf_level_1"

# level 2
rf_level2_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_2_94.07.pth"
name_rf_level2_s1 = "_seed_1_rf_level_2"

rf_level2_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_2_94.03.pth"
name_rf_level2_s2 = "_seed_2_rf_level_2"
# Level 3

rf_level3_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_3_92.38.pth"
name_rf_level3_s1 = "_seed_1_rf_level_3"

rf_level3_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_3_92.25.pth"
name_rf_level3_s2 = "_seed_2_rf_level_3"

# Level 4
rf_level4_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_4_90.66.pth"
name_rf_level4_s1 = "_seed_1_rf_level_4"
rf_level4_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_4_90.8.pth"
name_rf_level4_s2 = "_seed_2_rf_level_4"

files_names = [name_rf_level1_s1, name_rf_level1_s2, name_rf_level2_s1, name_rf_level2_s2, name_rf_level3_s1,
               name_rf_level3_s2, name_rf_level4_s1, name_rf_level4_s2]
files = [rf_level1_s1, rf_level1_s2, rf_level2_s1, rf_level2_s2, rf_level3_s1, rf_level3_s2, rf_level4_s1, rf_level4_s2]
level = [1, 1, 2, 2, 3, 3, 4, 4]
modelstypes = ["alternative"] * len(level)


def record_features_cifar10_model_pruned(architecture="resnet18", seed=1, modeltype="alternative", solution="",
                                         seed_name="_seed_1", rf_level=0, model=None):
    from feature_maps_utils import save_layer_feature_maps_for_batch

    cfg = omegaconf.DictConfig(
        {"architecture": architecture,
         "model_type": modeltype,
         "solution": solution,
         # "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         "dataset": "cifar10",
         "batch_size": 1,
         "num_workers": 0,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })
    resnet18_normal = model
    ################################# dataset cifar10 ###########################################################################

    from alternate_models.resnet import ResNet50_rf
    from torchvision.models import resnet18, resnet50
    if cfg.dataset == "cifar10":
        current_directory = Path().cwd()
        data_path = "/datasets"
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
            data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)

        # cifar10_train, cifar10_val = random_split(trainset, [45000, 5000])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=False,
                                                  num_workers=cfg.num_workers)
        # val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=cfg.batch_size, shuffle=True,
        #                                          num_workers=cfg.num_workers)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers)
    if cfg.dataset == "cifar100":
        if cfg.model_type == "alternative":
            resnet18_normal = ResNet50_rf(num_classes=100, rf_level=rf_level)
            resnet18_normal.load_state_dict(torch.load(cfg.solution)["net"])
        if cfg.model_type == "hub":
            resnet18_normal = resnet50()
            in_features = resnet18_normal.fc.in_features
            resnet18_normal.fc = torch.nn.Linear(in_features, 100)
            temp_dict = torch.load(cfg.solution)["net"]
            real_dict = {}
            for k, item in temp_dict.items():
                if k.startswith('module'):
                    new_key = k.replace("module.", "")
                    real_dict[new_key] = item
            resnet18_normal.load_state_dict(real_dict)
        current_directory = Path().cwd()
        data_path = ""
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "luis alfredo" == current_directory.owner() or "luis alfredo" in current_directory.__str__():
            data_path = "c:/users\luis alfredo\onedrive - university of leeds\phd\datasets\cifar100"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"

        transform_train = transforms.compose([
            transforms.randomcrop(32, padding=4),
            transforms.randomhorizontalflip(),
            transforms.totensor(),
            transforms.normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
        ])
        transform_test = transforms.compose([
            transforms.totensor(),
            transforms.normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
        ])

        trainset = torchvision.datasets.cifar100(root=data_path, train=True, download=True, transform=transform_train)

        trainloader = torch.utils.data.dataloader(trainset, batch_size=cfg.batch_size, shuffle=False,
                                                  num_workers=cfg.num_workers)
        testset = torchvision.datasets.cifar100(root=data_path, train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.dataloader(testset, batch_size=cfg.batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers)

    current_directory = Path().cwd()
    add_nobackup = ""
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        add_nobackup = "/nobackup/sclaam/"

    prefix_custom_test = Path(
        "{}features/{}/{}/{}/{}/".format(add_nobackup, cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    prefix_custom_test.mkdir(parents=True, exist_ok=True)
    ######################## now the pytorch implementation ############################################################
    maximun_samples = 2000
    resnet18_normal.cuda()
    o = 0
    for x, y in testloader:
        x = x.cuda()
        save_layer_feature_maps_for_batch(resnet18_normal, x, prefix_custom_test, seed_name=seed_name)
        # Path(file_prefix / "layer{}_features{}.npy".format(i, seed_name))

        print("{} batch out of {}".format(o, len(testloader)))
        if o == maximun_samples:
            break
        o += 1


def save_pruned_representations():
    cfg = omegaconf.DictConfig(
        {"architecture": "resnet50",
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": "cifar10",
         "batch_size": 128,
         "num_workers": 2,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })
    train, val, testloader = get_datasets(cfg)

    dense_accuracy_list = []
    pruned_accuracy_list = []

    for i in range(len(files)):
        state_dict_raw = torch.load(files[i])
        dense_accuracy_list.append(state_dict_raw["acc"])
        net = ResNet50_rf(num_classes=10, rf_level=level[i])
        net.load_state_dict(state_dict_raw["net"])
        prune_function(net, cfg)
        remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)
        record_features_cifar10_model_pruned("resnet50", seed_name="pruned_{}".format(files_names[i]),
                                             rf_level=level[i], model=net)


def similarity_comparisons():
    for i in range(len(files)):
        for j in range(i, len(files)):
            p_name1 = "pruned_{}".format(files_names[i])
            p_name2 = "pruned_{}".format(files_names[j])
            features_similarity_comparison_experiments(architecture="resnet50",
                                                       modeltype1="alternative",
                                                       modeltype2="alternative",
                                                       name1=p_name1,
                                                       name2=p_name2, filetype1="npy",
                                                       filetype2="npy")
            # filename = "similarity_experiments/{}_{}_V_{}_.txt".format("resnet50", p_name1, p_name2)
            #
            # np.savetxt(filename, similarity_for_networks, delimiter=",")


def gradient_flow_calculation(args):
    from flowandprune.imp_estimator import cal_grad
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": "resnet50",
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": exclude_layers

         })
    train, val, testloader = get_datasets(cfg)

    from torchvision.models import resnet18, resnet50
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

    gradient_flow_at_init_list = []
    pruned_accuracy_list = []
    files_names = []

    for i, name in enumerate(
            glob.glob("{}/{}_normal_{}_*_level_{}_initial_weights.pth".format(args.folder, args.model, args.dataset,
                                                                              args.RF_level))):
        state_dict_raw = torch.load(name)
        net.load_state_dict(state_dict_raw["net"])
        net.cuda()
        gradient_flow = cal_grad(net, trainloader=train)
        grad_vect = parameters_to_vector(gradient_flow)
        # hg_vect = parameters_to_vector(hg)
        norm_grad = torch.norm(grad_vect)
        gradient_flow_at_init_list.append(norm_grad.detach().cpu().numpy())
        file_name = os.path.basename(name)
        print(file_name)
        files_names.append(file_name)
    df = pd.DataFrame({"Name": files_names,
                       "Gradient Flow at init": gradient_flow_at_init_list,
                       })
    df.to_csv("GF_init_{}_{}_{}_summary.csv".format(args.model, args.RF_level, args.dataset), index=False)


def fine_tune_pruned_model_with_mask(pruned_model: nn.Module, dataLoader: torch.utils.data.DataLoader,
                                     testLoader: torch.utils.data.DataLoader,
                                     epochs=1,
                                     initial_flops=0, exclude_layers=[],
                                     fine_tune_exclude_layers=False, fine_tune_non_zero_weights=True,
                                     cfg=None, save_folder="", name=""):
    from main import get_mask
    from sparse_ensemble_utils import disable_bn, disable_all_except, disable_exclude_layers, mask_gradient
    from train_CIFAR10 import progress_bar

    optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.0001,
                                momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    grad_clip = 0
    # if cfg.gradient_cliping:
    #     grad_clip = 0.1
    # names, weights = zip(*get_layer_dict(pruned_model))

    mask_dict = get_mask(model=pruned_model)
    for n in exclude_layers:
        if n in list(mask_dict.keys()):
            mask_dict.pop(n)

    total_sparse_FLOPS = initial_flops
    # first_time = 1

    data, y = next(iter(dataLoader))
    # forward_pass_dense_flops, forward_pass_sparse_flops = flops(pruned_model, data)

    file_path = None

    pruned_model.cuda()
    pruned_model.train()
    # disable_bn(pruned_model)
    # if not fine_tune_exclude_layers:
    #     disable_exclude_layers(pruned_model, exclude_layers)
    # if not fine_tune_non_zero_weights:
    #     disable_all_except(pruned_model, exclude_layers)
    #
    criterion = nn.CrossEntropyLoss()

    pruned_model.train()
    train_loss = 0
    correct = 0
    total = 0
    best_acc = 0
    for epoch in range(epochs):
        #################################
        # Train
        #################################
        for batch_idx, (inputs, targets) in enumerate(dataLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = pruned_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Mask the gradient
            mask_gradient(pruned_model, mask_dict=mask_dict)
            # if grad_clip:
            #     nn.utils.clip_grad_value_(pruned_model.parameters(), grad_clip)

            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(100. * correct / total, correct, total)
            progress_bar(batch_idx, len(dataLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        #################################
        #    TEST
        #################################

        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testLoader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = pruned_model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, len(testLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': pruned_model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)
            if os.path.isfile('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc)):
                os.remove('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc))
            torch.save(state, '{}/{}_test_acc_{}.pth'.format(save_folder, name, acc))
            best_acc = acc

    return total_sparse_FLOPS


def pruning_fine_tuning_experiment(args):
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": "resnet50",
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
         "pruner": "global",
         "exclude_layers": exclude_layers

         })
    train, val, testloader = get_datasets(cfg)

    from torchvision.models import resnet18, resnet50
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

    dense_accuracy_list = []
    fine_tuned_accuracy = []
    folder_name = "{}/pruned/{}".format(args.folder, args.pruning_rate)
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    # new_folder = "{}/pruned/{}".format(args.folder, args.pruning_rate)

    #
    # for i, name in enumerate(
    #         glob.glob("{}/{}_normal_{}_*_level_{}_test_acc_*.pth".format(args.folder, args.model, args.dataset,
    #                                                                      f"{args.RF_level}{args.name}"))):

    state_dict_raw = torch.load("{}/{}".format(args.folder, args.solution))
    dense_accuracy_list.append(state_dict_raw["acc"])
    net.load_state_dict(state_dict_raw["net"])
    prune_function(net, cfg)
    remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)

    file_name = args.solution
    index_until_test = file_name.index("test_acc")
    base_name = file_name[:index_until_test]

    # Strings in between _

    fine_tune_pruned_model_with_mask(net, dataLoader=train, testLoader=testloader, epochs=20,
                                     exclude_layers=cfg.exclude_layers, cfg=cfg, save_folder=folder_name,
                                     name=base_name)

    # if os.path.isfile('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc)):
    #     os.remove('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc))
    #
    # strings_in_between = re.findall("(?<=\_)(.*?)(?=\_)", file_name)


def main(args):
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": "resnet50",
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": exclude_layers

         })
    train, val, testloader = get_datasets(cfg)

    from torchvision.models import resnet18, resnet50
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

    dense_accuracy_list = []
    pruned_accuracy_list = []
    files_names = []

    for i, name in enumerate(
            glob.glob("{}/{}_normal_{}_*_level_{}_test_acc_*.pth".format(args.folder, args.model, args.dataset,
                                                                         f"{args.RF_level}{args.name}"))):
        state_dict_raw = torch.load(name)
        dense_accuracy_list.append(state_dict_raw["acc"])
        net.load_state_dict(state_dict_raw["net"])
        prune_function(net, cfg)
        remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)
        pruned_accuracy = test(net, use_cuda=True, testloader=testloader, verbose=0)
        pruned_accuracy_list.append(pruned_accuracy)
        weight_names, weights = zip(*get_layer_dict(net))
        zero_number = lambda w: (torch.count_nonzero(w == 0) / w.nelement()).cpu().numpy()
        pruning_rates_per_layer = list(map(zero_number, weights))
        seed_from_file = re.findall("_[0-9]_", name)[0].replace("_", "")
        df2 = pd.DataFrame({"layer_names": weight_names, "pr": pruning_rates_per_layer})
        df2.to_csv(
            "{}_level_{}_seed_{}_{}_pruning_rates.csv".format(args.model, args.RF_level, seed_from_file, args.dataset),
            index=False)
        file_name = os.path.basename(name)
        print(file_name)
        files_names.append(file_name)

    df = pd.DataFrame({"Name": files_names,
                       "Dense Accuracy": dense_accuracy_list,
                       "Pruned Accuracy": pruned_accuracy_list,
                       })
    df.to_csv("RF_{}_{}_{}_summary.csv".format(args.model, args.RF_level, args.dataset), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='One shot pruning statistics')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--type', default="normal", type=str, help='Type of implementation [normal,official]')
    parser.add_argument('--RF_level', default=4, type=int, help='Receptive field level')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', default="cifar10", type=str, help='Dataset to use [cifar10,tiny_imagenet616gg]')
    parser.add_argument('--model', default="resnet18", type=str, help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where saved models are')
    parser.add_argument('--name', default="", type=str, help='Name of the file')
    parser.add_argument('--solution', default="", type=str, help='Solution to use')
    parser.add_argument('--pruning_rate', default=0.9, type=float, help='Pruning rate')
    args = parser.parse_args()
    # main(args)
    pruning_fine_tuning_experiment(args)
    # gradient_flow_calculation(args)
    # save_pruned_representations()
    # similarity_comparisons()
