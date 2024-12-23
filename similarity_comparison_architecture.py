import pandas as pd

print("I'm about to begin the similarity comparison")
import omegaconf
from pathlib import Path
import re
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import numpy as np
import time
from feature_maps_utils import load_layer_features, save_layer_feature_maps_for_batch
from torch.utils.data import random_split
import argparse
from alternate_models import *

# matplotlib.rcParams['figure.figsize'] = [18, 12]
# training hyperparameters
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 10 ** -2
BATCH_SIZE = 512
EPOCHS = 25
# contour plot resolution
STEPS = 40
# Setting random seeds
manual_seed_generator = torch.manual_seed(2809)
np.random.seed(2809)
import random

random.seed(2809)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Same as linear regression!
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


class WrapperVisionModel(nn.Module):
    def __init__(self, net, input_dim, output_dim):
        super(WrapperVisionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        def new_forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out1 = self.layer1(out)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            return out1.view(out1.size(0), -1), out2.view(out2.size(0), -1), out3.view(out.size(0), -1), out3.view(
                out.size(0), -1), out4.view(out.size(0), -1)

        net.__setattr__("fc2", nn.Linear(256, 10))
        net.forward = new_fowrard.__get__(net)  # bind method

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


def record_features_model_dataset(args):
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

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

        cifar10_train, cifar10_val = random_split(trainset, [len(trainset) - args.eval_size, args.eval_size])

        trainloader = torch.utils.data.DataLoader(
            cifar10_train, batch_size=128, shuffle=True, num_workers=args.num_workers)

        valloader = torch.utils.data.DataLoader(
            cifar10_val, batch_size=128, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    if args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform_train)

        cifar10_train, cifar10_val = random_split(trainset, [len(trainset) - 5000, 5000])

        trainloader = torch.utils.data.DataLoader(
            cifar10_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        valloader = torch.utils.data.DataLoader(
            cifar10_val, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

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
                                                                                      batch_size, args.num_workers,
                                                                                      valsize=args.eval_size,
                                                                                      testsize=args.eval_size,
                                                                                      shuffle_val=False,
                                                                                      shuffle_test=False, )
        else:

            from test_imagenet import load_small_imagenet
            trainloader, valloader, testloader = load_small_imagenet(
                {"traindir": data_path + "/small_imagenet/train", "valdir": data_path + "/small_imagenet/val",
                 "num_workers": args.num_workers, "batch_size": batch_size, "resolution": args.input_resolution},
                val_size=args.eval_size, test_size=args.eval_size, shuffle_val=False, shuffle_test=False,
                random_split_generator=manual_seed_generator, seed_worker=seed_worker)

    from torchvision.models import resnet18, resnet50

    if args.model == "resnet18":
        if args.modeltype1 == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, RF_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, RF_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "small_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet50":
        if args.modeltype1 == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level)
        if args.modeltype1 == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.modeltype1 == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg19":
        if args.modeltype1 == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, RF_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, RF_level=args.RF_level)

        if args.modeltype1 == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "small_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet24":
        if args.modeltype1 == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=args.RF_level)
        if args.modeltype1 == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.modeltype1))
    if args.model == "resnet_small":
        if args.modeltype1 == "normal" and args.dataset == "cifar10":
            net = small_ResNet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)
        if args.modeltype1 == "normal" and args.dataset == "cifar100":
            net = small_ResNet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width)
        if args.modeltype1 == "normal" and args.dataset == "tiny_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.modeltype1 == "normal" and args.dataset == "small_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.modeltype1 == "pytorch" and args.dataset == "cifar10":
            raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.modeltype1 == "pytorch" and args.dataset == "cifar100":
            raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)

    ###########################################################################
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

    net = net.to(device)
    net.eval()

    prefix_custom_test = Path(
        "{}/{}/{}/{}/{}/{}/".format(args.folder, args.dataset, args.model, args.RF_level, args.modeltype1, "test"))
    prefix_custom_test.mkdir(parents=True, exist_ok=True)
    ######################## now the pytorch implementation ############################################################
    maximun_samples = args.eval_size
    # seed_name= "_samples_{}_{}".format(maximun_samples,args.seedname1)
    net.cuda()
    o = 0
    for x, y in testloader:
        x = x.cuda()
        save_layer_feature_maps_for_batch(net, x, args.batch_size, prefix_custom_test, seed_name=args.seedname1)

        print("{} batch out of {}".format(o, len(testloader)))

        o += 1


def record_features_cifar10_model(architecture="resnet18", seed=1, modeltype="alternative", solution="",
                                  seed_name="_seed_1", rf_level=0):
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
    ################################# dataset cifar10 ###########################################################################

    from alternate_models.resnet import ResNet50_rf
    from alternate_models.vgg import VGG_RF
    from torchvision.models import resnet18, resnet50
    if cfg.dataset == "cifar10":
        if cfg.architecture == "resnet50":
            if cfg.model_type == "alternative":
                net = ResNet50_rf(num_classes=10, rf_level=rf_level)
                if solution:
                    net.load_state_dict(torch.load(cfg.solution)["net"])
            if cfg.model_type == "hub":
                net = resnet50()
                in_features = net.fc.in_features
                net.fc = torch.nn.Linear(in_features, 10)
                if solution:
                    temp_dict = torch.load(cfg.solution)["net"]
                    real_dict = {}
                    for k, item in temp_dict.items():
                        if k.startswith('module'):
                            new_key = k.replace("module.", "")
                            real_dict[new_key] = item
                    net.load_state_dict(real_dict)

        if cfg.architecture == "vgg19":

            if cfg.model_type == "alternative":
                net = VGG_RF("VGG19_rf", num_classes=10, RF_level=rf_level)

        current_directory = Path().cwd()
        data_path = "/datasets"
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
            data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"
        # data_path = "datasets" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
        #                                                                            "University of Leeds\PhD\Datasets\CIFAR10"

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
        if cfg.architecture == "resnet50":
            if cfg.model_type == "alternative":

                net = ResNet50_rf(num_classes=100, rf_level=rf_level)
                if solution:
                    net.load_state_dict(torch.load(cfg.solution)["net"])
            if cfg.model_type == "hub":
                net = resnet50()
                in_features = net.fc.in_features
                net.fc = torch.nn.Linear(in_features, 100)
                if solution:
                    temp_dict = torch.load(cfg.solution)["net"]
                    real_dict = {}
                    for k, item in temp_dict.items():
                        if k.startswith('module'):
                            new_key = k.replace("module.", "")
                            real_dict[new_key] = item
                    net.load_state_dict(real_dict)

        if cfg.architecture == "vgg19":

            if cfg.model_type == "alternative":
                net = VGG_RF("VGG19_rf", num_classes=100, rf_level=rf_level)
                if solution:
                    net.load_state_dict(torch.load(cfg.solution)["net"])

        current_directory = Path().cwd()
        data_path = ""
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "luis alfredo" == current_directory.owner() or "luis alfredo" in current_directory.__str__():
            data_path = "c:/users\luis alfredo\onedrive - university of leeds\phd\datasets\cifar100"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
        ])

        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=False,
                                                  num_workers=cfg.num_workers)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers)

    current_directory = Path().cwd()
    add_nobackup = ""
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        add_nobackup = "/nobackup/sclaam/"

    # prefix_custom_train = Path(
    #     "{}features/{}/{}/{}/{}/".format(add_nobackup, cfg.dataset, cfg.architecture, cfg.model_type, "train"))
    prefix_custom_test = Path(
        "{}features/{}/{}/{}/{}/".format(add_nobackup, cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    prefix_custom_test.mkdir(parents=True, exist_ok=True)
    ######################## now the pytorch implementation ############################################################
    maximun_samples = 1000
    net.cuda()
    o = 0
    for x, y in testloader:

        x = x.cuda()
        save_layer_feature_maps_for_batch(net, x, prefix_custom_test, seed_name=seed_name)

        print("{} batch out of {}".format(o, len(testloader)))
        if o == maximun_samples:
            break
        o += 1


def save_features_for_logistic(architecture="resnet18", seed=1, modeltype="alternative", solution="",
                               seed_name="_seed_1", rf_level=0, train=1):
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
    ################################# dataset cifar10 ###########################################################################

    from alternate_models.resnet import ResNet50_rf
    from alternate_models.vgg import VGG_RF
    from torchvision.models import resnet18, resnet50
    if cfg.dataset == "cifar10":
        # if cfg.architecture == "resnet50":
        #     if cfg.model_type == "alternative":
        #         net = ResNet50_rf(num_classes=10, rf_level=rf_level)
        #         if solution:
        #             net.load_state_dict(torch.load(cfg.solution)["net"])
        #     if cfg.model_type == "hub":
        #         net = resnet50()
        #         in_features = net.fc.in_features
        #         net.fc = torch.nn.Linear(in_features, 10)
        #         if solution:
        #             temp_dict = torch.load(cfg.solution)["net"]
        #             real_dict = {}
        #             for k, item in temp_dict.items():
        #                 if k.startswith('module'):
        #                     new_key = k.replace("module.", "")
        #                     real_dict[new_key] = item
        #             net.load_state_dict(real_dict)
        #
        # if cfg.architecture == "vgg19":
        #
        #     if cfg.model_type == "alternative":
        #         net = VGG_RF("VGG19_rf", num_classes=10, rf_level=rf_level)
        #
        current_directory = Path().cwd()
        data_path = "/datasets"
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
            data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"
        # data_path = "datasets" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
        #                                                                            "University of Leeds\PhD\Datasets\CIFAR10"

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

        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_test)

        # cifar10_train, cifar10_val = random_split(trainset, [45000, 5000])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=False,
                                                  num_workers=cfg.num_workers)
        # val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=cfg.batch_size, shuffle=True,
        #                                          num_workers=cfg.num_workers)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers)
    if cfg.dataset == "cifar100":
        if cfg.architecture == "resnet50":
            if cfg.model_type == "alternative":

                net = ResNet50_rf(num_classes=100, rf_level=rf_level)
                if solution:
                    net.load_state_dict(torch.load(cfg.solution)["net"])
            if cfg.model_type == "hub":
                net = resnet50()
                in_features = net.fc.in_features
                net.fc = torch.nn.Linear(in_features, 100)
                if solution:
                    temp_dict = torch.load(cfg.solution)["net"]
                    real_dict = {}
                    for k, item in temp_dict.items():
                        if k.startswith('module'):
                            new_key = k.replace("module.", "")
                            real_dict[new_key] = item
                    net.load_state_dict(real_dict)

        if cfg.architecture == "vgg19":

            if cfg.model_type == "alternative":
                net = VGG_RF("VGG19_rf", num_classes=100, rf_level=rf_level)
                if solution:
                    net.load_state_dict(torch.load(cfg.solution)["net"])

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
    prefix = None
    if train:
        prefix = prefix_custom_train = Path(
            "{}features/{}/{}/{}/{}/".format(add_nobackup, cfg.dataset, cfg.architecture, cfg.model_type, "train"))
    else:
        prefix = prefix_custom_test = Path(
            "{}features/{}/{}/{}/{}/".format(add_nobackup, cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    prefix.mkdir(parents=True, exist_ok=True)
    ######################## now the pytorch implementation ############################################################
    maximun_samples = 20000
    # net.cuda()
    o = 0
    dataloader = None
    if train:
        dataloader = trainloader
    else:
        dataloader = testloader
    save_y = []
    for x, y in dataloader:
        save_y.append(int(y.cpu().detach().item()))
        x = x.cuda()
        save_layer_feature_maps_for_batch(net, x, prefix, seed_name=seed_name)
        # Path(file_prefix / "layer{}_features{}.npy".format(i, seed_name))

        print("{} batch out of {}".format(o, len(dataloader)))
        if o == maximun_samples:
            break
        o += 1
    if train:
        np.savetxt("{}_train_20k.txt".format(cfg.dataset), save_y, delimiter=",")
    else:
        np.savetxt("{}_test_20k.txt".format(cfg.dataset), save_y, delimiter=",")


def test_function(model, test_loader, epoch, train=False):
    print_string = "Test"
    if train:
        print_string = "Train"
    # Calculate Accuracy
    correct = 0
    total = 0
    # Iterate through test dataset
    for features, labels in test_loader:
        # Load images to a Torch Variable
        images = features.cuda()
        labels = labels.type(torch.LongTensor)
        labels = labels.cuda()

        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum()

    accuracy = 100 * correct / total

    # Print Loss
    print('Epoch {}: {} accuracy {}.'.format(epoch, print_string, accuracy))
    return accuracy


def experiment_training_logistic_function(architecture="resnet18", modeltype1="alternative",
                                          modeltype2="alternative", name1="_seed_1", name2="_seed_2",
                                          filetype1="npy", filetype2="txt", rf_level=1, layer_index=0):
    cfg = omegaconf.DictConfig(
        {"architecture": architecture,
         "model_type": modeltype1,
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         "dataset": "cifar10",
         "batch_size": 128,
         "num_workers": 2,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })

    prefix_modeltype1_train = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "train"))
    prefix_modeltype1_test = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    train_logistic_on_specific_layer(cfg.architecture, rf_level=rf_level, prefix_train=prefix_modeltype1_train,
                                     prefix_test=prefix_modeltype1_test, layer_index=layer_index, seed_name=name1,
                                     type=filetype1)


def train_logistic_on_specific_layer(model_name, rf_level, prefix_train, prefix_test, layer_index, seed_name="",
                                     type="npy"):
    print("Type used for loading:{}".format(type))
    train_x = load_layer_features(prefix_train, layer_index, seed_name, type)
    samples, n_features = train_x.shape
    train_y = np.loadtxt("cifar10_train_20k.txt")
    test_x = load_layer_features(prefix_test, layer_index, seed_name, type)
    test_y = np.loadtxt("cifar10_test_20k.txt")
    criterion = nn.CrossEntropyLoss()
    model = LogisticRegressionModel(n_features, 10)
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=0.9, weight_decay=5e-4)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=0)

    iter = 0
    num_epochs = 400
    accuracy_list = []
    epoch_list = []
    best_test_accuracy = 0
    model.cuda()
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_loader):
            # Load images as Variable
            labels = labels.type(torch.LongTensor)
            features = features.requires_grad_()
            images = features.cuda()
            labels = labels.cuda()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1
        train_accuracy = test_function(model, train_loader, epoch, train=True)
        test_accuracy = test_function(model, test_loader, epoch)
        epoch_list.append(epoch)
        accuracy_list.append(test_accuracy)
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
    print("Best test Accuracy: {}".format(best_test_accuracy))
    df = pd.DataFrame(
        {
            "Epoch": epoch_list,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
        }

    )
    df.to_csv(
        "logistic_{}_{}_layer_{}_{}.csv".format(model_name, rf_level, layer_index, best_test_accuracy),
        index=False)


def features_similarity_comparison_experiments(architecture="resnet18", modeltype1="alternative",
                                               modeltype2="alternative", name1="_seed_1", name2="_seed_2",
                                               filetype1="txt", filetype2="txt"):
    cfg = omegaconf.DictConfig(
        {"architecture": architecture,
         "model_type": modeltype1,
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         "dataset": "cifar10",
         "batch_size": 128,
         "num_workers": 2,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })

    prefix_custom_train = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "train"))
    prefix_modeltype1_test = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    cfg.model_type = modeltype2
    prefix_modeltype2_test = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))

    ##### -1 beacuse I dont have the linear layer here
    # number_of_layers = int(re.findall(r"\d+", cfg.architecture)[0]) - 1
    number_of_layers = 0
    if cfg.architecture == "resnet50":
        number_of_layers = 49
    if cfg.architecture == "vgg19":
        number_of_layers = 16
    similarity_for_networks = representation_similarity_analysis(prefix_modeltype1_test, prefix_modeltype2_test,
                                                                 number_layers=number_of_layers, name1=name1,
                                                                 name2=name2, type1=filetype1, type2=filetype2,
                                                                 use_device="cuda")
    filename = "similarity_experiments/{}_{}_V_{}_.txt".format(cfg.architecture, name1, name2)

    np.savetxt(filename, similarity_for_networks, delimiter=",")


def features_similarity_comparison_experiments_model_combination(args):
    # architecture="resnet18", modeltype1="alternative",
    #                                                              modeltype2="alternative", name1="_seed_1",
    #                                                              name2="_seed_2",
    #                                                              filetype1="txt", filetype2="txt"):
    # cfg = omegaconf.DictConfig(
    #     {"architecture": architecture,
    #      "model_type": modeltype1,
    #      "solution": "trained_models/cifar10/resnet50_cifar10.pth",
    #      "dataset": "cifar10",
    #      "batch_size": 128,
    #      "num_workers": 2,
    #      "amount": 0.9,
    #      "noise": "gaussian",
    #      "sigma": 0.005,
    #      "pruner": "global",
    #      "exclude_layers": ["conv1", "linear"]
    #
    #      })

    prefix_modeltype1_test = Path(
        "{}/{}/{}/{}/{}/{}/".format(args.folder, args.dataset, args.model, args.RF_level, args.modeltype1, "test"))
    prefix_modeltype2_test = Path(
        "{}/{}/{}/{}/{}/{}/".format(args.folder, args.dataset, args.model, args.RF_level2, args.modeltype2, "test"))

    ##### -1 beacuse I dont have the linear layer here
    # number_of_layers = int(re.findall(r"\d+", cfg.architecture)[0]) - 1
    number_of_layers = 0
    if args.model == "resnet50":
        number_of_layers = 49
    if args.model == "vgg19":
        number_of_layers = 16
    if args.model == "resnet_small":
        number_of_layers = 7

    similarity_for_networks = representation_similarity_analysis(prefix_modeltype1_test, prefix_modeltype2_test,
                                                                 number_layers=number_of_layers, name1=args.seedname1,
                                                                 name2=args.seedname2, type1=args.filetype1,
                                                                 type2=args.filetype2,
                                                                 use_device=args.device,
                                                                 subtract_mean=args.subtract_mean)

    subtracted_mean_string = "mean_subtracted" if args.subtract_mean else "mean_not_subtracted"

    filename = "similarity_experiments/{}_RF_{}_{}_V_RF_{}_{}_{}.txt".format(args.model, args.RF_level, args.seedname1,
                                                                             args.RF_level2, args.seedname2,
                                                                             subtracted_mean_string)

    np.savetxt(filename, similarity_for_networks, delimiter=",")


@torch.no_grad()
def representation_similarity_analysis(prefix1, prefix2, number_layers, name1="", name2="", type1="txt", type2="txt",
                                       use_device="cuda", subtract_mean=1):
    from CKA_similarity.CKA import CudaCKA, CKA
    print("Use_device: {}".format(use_device))
    if use_device == "cuda":
        kernel = CudaCKA("cuda")
        similarity_matrix = torch.zeros((number_layers, number_layers), device=use_device)

    if use_device == "cpu":
        similarity_matrix = np.zeros((number_layers, number_layers))
        kernel = CKA()
    #### because the similiarity is a simetrical
    for i in range(number_layers):
        if use_device == "cuda":
            layer_i = torch.tensor(load_layer_features(prefix1, i, name=name1, type=type1))
        if use_device == "cpu":
            layer_i = load_layer_features(prefix1, i, name=name1, type=type1)
        for j in range(i, number_layers):

            if use_device == "cuda":
                t0 = time.time()
                print("We are in row {} and colum {}".format(i, j))
                layer_j = torch.tensor(load_layer_features(prefix2, j, name=name2, type=type2))
                t1 = time.time()
                print("Time of loading layer: {}".format(t1 - t0))
                layeri_cuda = layer_i.cuda()
                layerj_cuda = layer_j.cuda()
                if subtract_mean:
                    layeri_cuda = layeri_cuda - torch.mean(layeri_cuda, dtype=torch.float, dim=0)
                    layerj_cuda = layerj_cuda - torch.mean(layerj_cuda, dtype=torch.float, dim=0)

                t0 = time.time()
                similarity_matrix[i, j] = kernel.linear_CKA(layeri_cuda.float(), layerj_cuda.float())
                t1 = time.time()

                print("Time for linear kernel: {}".format(t1 - t0))
                del layeri_cuda
                del layerj_cuda
                torch.cuda.empty_cache()

            if use_device == "cpu":

                t0 = time.time()
                print("We are in row {} and colum {}".format(i, j))
                # layer_i = load_layer_features(prefix1, i, name=name1)
                # print("Time loading layer i {}".format(time.time() - t0))
                tm = time.time()
                layer_j = load_layer_features(prefix2, j, name=name2, type=type2)
                print("Time loading layer j {}".format(time.time() - tm))

                if subtract_mean:
                    layer_i = layer_i - np.mean(layer_i, dtype=float, axis=0)
                    layer_j = layer_j - np.mean(layer_j, dtype=float, axis=0)

                similarity_matrix[i, j] = kernel.linear_CKA(layer_i, layer_j)
                t1 = time.time()
                print("Time of loading + linear kernel: {}".format(t1 - t0))
                # del layer_i
                # del layer_i
                del layer_j

    # network1 =
    if use_device == "cuda":
        simetric_similarity = similarity_matrix.add(similarity_matrix.T)
        simetric_similarity[range(number_layers), range(number_layers)] *= 1 / 2
        return simetric_similarity.detach().cpu().numpy()
    if use_device == "cpu":
        simetric_similarity = similarity_matrix + np.transpose(similarity_matrix)
        simetric_similarity[range(number_layers), range(number_layers)] *= 1 / 2
        return simetric_similarity


def find_different_groups(architecture="resnet18", seed=1, modeltype="alternative", solution="",
                          seed_name="_seed_1", rf_level=0):
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
    ################################# dataset cifar10 ###########################################################################

    from alternate_models.resnet import ResNet50_rf
    from torchvision.models import resnet18, resnet50
    if cfg.dataset == "cifar10":
        if cfg.model_type == "alternative":
            net = ResNet50_rf(num_classes=10, rf_level=rf_level)
            if solution:
                net.load_state_dict(torch.load(cfg.solution)["net"])
        if cfg.model_type == "hub":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = torch.nn.Linear(in_features, 10)
            if solution:
                temp_dict = torch.load(cfg.solution)["net"]
                real_dict = {}
                for k, item in temp_dict.items():
                    if k.startswith('module'):
                        new_key = k.replace("module.", "")
                        real_dict[new_key] = item
                net.load_state_dict(real_dict)

        current_directory = Path().cwd()
        data_path = "/datasets"
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
            data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"
        # data_path = "datasets" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
        #                                                                            "University of Leeds\PhD\Datasets\CIFAR10"

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
            net = ResNet50_rf(num_classes=100, rf_level=rf_level)
            net.load_state_dict(torch.load(cfg.solution)["net"])
        if cfg.model_type == "hub":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = torch.nn.Linear(in_features, 100)
            temp_dict = torch.load(cfg.solution)["net"]
            real_dict = {}
            for k, item in temp_dict.items():
                if k.startswith('module'):
                    new_key = k.replace("module.", "")
                    real_dict[new_key] = item
            net.load_state_dict(real_dict)
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

    # prefix_custom_train = Path(
    #     "{}features/{}/{}/{}/{}/".format(add_nobackup, cfg.dataset, cfg.architecture, cfg.model_type, "train"))
    prefix_custom_test = Path(
        "{}features/{}/{}/{}/{}/".format(add_nobackup, cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    prefix_custom_test.mkdir(parents=True, exist_ok=True)
    ######################## now the pytorch implementation ############################################################
    maximun_samples = 2000
    net.cuda()
    o = 0
    for x, y in testloader:
        x = x.cuda()
        save_layer_feature_maps_for_batch(net, x, prefix_custom_test, seed_name=seed_name)
        # Path(file_prefix / "layer{}_features{}.npy".format(i, seed_name))

        print("{} batch out of {}".format(o, len(testloader)))
        if o == maximun_samples:
            break
        o += 1


def describe_statistics_of_layer_representations(architecture="resnet18", modeltype1="alternative",
                                                 modeltype2="alternative", name1="_seed_1", name2="_seed_2",
                                                 filetype1="txt", filetype2="txt"):
    from CKA_similarity.CKA import CKA
    cfg = omegaconf.DictConfig(
        {"architecture": architecture,
         "model_type": modeltype1,
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         "dataset": "cifar10",
         "batch_size": 128,
         "num_workers": 2,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })

    prefix_custom_train = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "train"))
    prefix_modeltype1_test = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    cfg.model_type = modeltype2
    prefix_modeltype2_test = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))

    ##### -1 beacuse I dont have the linear layer here
    # number_of_layers = int(re.findall(r"\d+", cfg.architecture)[0]) - 1
    number_of_layers = 0
    if cfg.architecture == "resnet50":
        number_of_layers = 49
    if cfg.architecture == "vgg19":
        number_of_layers = 16
    feature_length_list = []
    layer_index_list = []
    max_all_rep_list = []
    mean_all_rep_list = []

    feature_length_minus_mean_list = []
    layer_index_minus_mean_list = []
    max_all_rep_minus_mean_list = []
    mean_all_rep_minus_mean_list = []
    kernel = CKA()
    for i in range(number_of_layers):
        # if use_device == "cuda":
        layer_feature = load_layer_features(prefix_modeltype1_test, i, name=name1, type=filetype1)
        mean_per_feature = np.mean(layer_feature, axis=0)
        sd_per_feature = np.std(layer_feature, axis=0)

        # layer_i = pd.DataFrame(layer_feature)
        print("##### Layer {} ###########".format(i))
        num_features = len(mean_per_feature)
        print("Feature length: {}".format(num_features))
        print("How many 0 are in the mean vector per feature? : {}".format(
            num_features - np.count_nonzero(mean_per_feature)))
        print("How many 0 are in the standard deviation vector per feature?: {}".format(
            num_features - np.count_nonzero(sd_per_feature)))
        print("Mean of the whole features over all samples: {}".format(np.mean(mean_per_feature)))
        print("SD of the whole features over all samples: {}".format(np.std(layer_feature)))
        print("Maximum of the whole features over all samples: {}".format(np.max(layer_feature)))
        feature_length_list.append(num_features)
        layer_index_list.append(i)
        max_all_rep_list.append(np.max(layer_feature))
        mean_all_rep_list.append(np.mean(mean_per_feature))

        print("##### Minus mean Layer {} ###########".format(i))
        layer_feature = load_layer_features(prefix_modeltype1_test, i, name=name1, type=filetype1)
        mean_per_feature = np.mean(layer_feature, axis=0)
        layer_feature = layer_feature - mean_per_feature
        mean_per_feature = np.mean(layer_feature, axis=0)
        sd_per_feature = np.std(layer_feature, axis=0)
        # layer_i = pd.DataFrame(layer_feature)
        num_features = len(mean_per_feature)
        print("Feature length: {}".format(num_features))
        print("How many 0 are in the mean vector per feature? : {}".format(
            num_features - np.count_nonzero(mean_per_feature)))
        print("How many 0 are in the standard deviation vector per feature?: {}".format(
            num_features - np.count_nonzero(sd_per_feature)))
        print("Mean of the whole features over all samples: {}".format(np.mean(mean_per_feature)))
        print("SD of the whole features over all samples: {}".format(np.std(layer_feature)))
        print("Maximum of the whole features over all samples: {}".format(np.max(layer_feature)))
        feature_length_minus_mean_list.append(num_features)
        layer_index_minus_mean_list.append(i)
        max_all_rep_minus_mean_list.append(np.max(layer_feature))
        mean_all_rep_minus_mean_list.append(np.mean(mean_per_feature))
        #
        # print("##### Minus mean and kernel centering Layer {} ###########".format(i))
        # layer_feature = load_layer_features(prefix_modeltype1_test, i, name=name1, type=filetype1)
        # mean_per_feature = np.mean(layer_feature, axis=0)
        # layer_feature = layer_feature-mean_per_feature
        # layer_feature
        # sd_per_feature = np.std(layer_feature, axis=0)
        # # layer_i = pd.DataFrame(layer_feature)
        # num_features = len(mean_per_feature)
        # print("Feature length: {}".format(num_features))
        # print("How many 0 are in the mean vector per feature? : {}".format(
        #     num_features - np.count_nonzero(mean_per_feature)))
        # print("How many 0 are in the standard deviation vector per feature?: {}".format(
        #     num_features - np.count_nonzero(sd_per_feature)))
        # print("Mean of the whole features over all samples: {}".format(np.mean(mean_per_feature)))
        # print("SD of the whole features over all samples: {}".format(np.std(layer_feature)))
        # print("Maximum of the whole features over all samples: {}".format(np.max(layer_feature)))
        # layer_i.describe()
        #      feature_length_minus_mean_list = []
        # layer_index_minus_mean_list = []
        # max_all_rep_minus_mean_list = []
        # mean_all_rep_minus_mean_list = []
        df1 = pd.DataFrame(
            {"Representation Length": feature_length_list,
             "Representations Index": layer_index_list,
             "Maximum Value of Representations": max_all_rep_list,
             "Mean Value of Representations": mean_all_rep_list
             }
        )
        df2 = pd.DataFrame(
            {"Representation Length": feature_length_minus_mean_list,
             "Representations Index": layer_index_minus_mean_list,
             "Maximum Value of Representations": max_all_rep_minus_mean_list,
             "Mean Value of Representations": mean_all_rep_minus_mean_list
             }

        )
        df1.to_csv("Representations_statistics_{}_{}_{}_{}_one_shot_summary.csv".format(args.model, args.RF_level,
                                                                                        args.dataset,
                                                                                        args.pruning_rate),
                   index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Similarity experiments')
    # parser.add_argument('-arch', '--architecture', type=str, default="resnet18", help='Architecture for analysis',
    #                     required=True)

    parser.add_argument('-s', '--solution', type=str, default="", help='',
                        required=False)

    parser.add_argument('-sn1', '--seedname1', type=str, default="", help='',
                        required=True)
    parser.add_argument('-sn2', '--seedname2', type=str, default="", help='',
                        required=False)
    parser.add_argument('-li', '--layer_index', type=int, default=0, help='',
                        required=False)
    parser.add_argument('-e', '--experiment', type=int, default=1, help='',
                        required=False)
    parser.add_argument('-mt1', '--modeltype1', type=str, default="alternative", help='',
                        required=False)
    parser.add_argument('-mt2', '--modeltype2', type=str, default="alternative", help='',
                        required=False)
    parser.add_argument('-ft1', '--filetype1', type=str, default="npy", help='',
                        required=False)
    parser.add_argument('-ft2', '--filetype2', type=str, default="npy", help='',
                        required=False)
    parser.add_argument('-t', '--train', type=int, default=1, help='',
                        required=False)

    parser.add_argument('--RF_level', default=3, type=int, help='Receptive field of model 1')
    parser.add_argument('--RF_level2', default=3, type=int, help='Receptive field of model 2')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', default="cifar10", type=str, help='Dataset to use [cifar10,cifar100]')
    parser.add_argument('--model', default="resnet50", type=str, help='Architecture of model [resnet18,resnet50]')

    parser.add_argument('--name', '-n', default="no_name", help='name of the loss files, usually the seed name')
    parser.add_argument('--eval_set', '-es', default="val", help='On which set to performa the calculations')
    parser.add_argument('--folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where the output of the algorithm is going to be saved')
    parser.add_argument('--data_folder', default="/nobackup/sclaam/data", type=str,
                        help='Location of the dataset', required=True)
    parser.add_argument('--width', default=1, type=int, help='Width of the model')
    parser.add_argument('--eval_size', default=1000, type=int, help='How many images to use in the calculation')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size for loading data')
    parser.add_argument('--input_resolution', default=224, type=int,
                        help='Input Resolution for Small ImageNet')
    parser.add_argument('--ffcv', action='store_true', help='Use FFCV loaders')

    parser.add_argument('--ffcv_train',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv",
                        type=str, help='FFCV train dataset')

    parser.add_argument('--ffcv_val',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv",
                        type=str, help='FFCV val dataset')
    parser.add_argument('--device', default="cpu", type=str, help='Which device to perform the matrix multiplication')
    parser.add_argument('--subtract_mean', default=1, type=int, help='Subtract mean of representations')
    #
    args = parser.parse_args()
    # features_similarity_comparison_experiments(args["architecture"])

    # rf_level2_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_2_94.07.pth"
    # name_rf_level2_s1 = "_seed_1_rf_level_2"
    # rf_level2_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_2_94.03.pth"
    # name_rf_level2_s2 = "_seed_2_rf_level_2"
    # rf_level4_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_4_90.66.pth"
    # name_rf_level4_s1 = "_seed_1_rf_level_4"
    # rf_level4_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_4_90.8.pth"
    # name_rf_level4_s2 = "_seed_2_rf_level_4"
    if args.experiment == 1:
        # record_features_cifar10_model(args["architecture"], modeltype=args["modeltype1"], solution=args["solution"],
        #                               seed_name=args["seedname1"])
        record_features_model_dataset(args)
    #
    if args.experiment == 2:
        # features_similarity_comparison_experiments(architecture=args["architecture"], modeltype1=args["modeltype1"],
        #                                            modeltype2=args["modeltype2"], name1=args["seedname1"],
        #                                            name2=args["seedname2"], filetype1=args["filetype1"],
        #                                            filetype2=args["filetype2"])
        features_similarity_comparison_experiments_model_combination(args)
    if args.experiment == 3:
        save_features_for_logistic(args["architecture"], modeltype=args["modeltype1"], solution=args["solution"],
                                   seed_name=args["seedname1"], train=args["train"])
    if args.experiment == 4:
        describe_statistics_of_layer_representations(architecture=args["architecture"], modeltype1=args["modeltype1"],
                                                     modeltype2=args["modeltype2"], name1=args["seedname1"],
                                                     name2=args["seedname2"], filetype1=args["filetype1"],
                                                     filetype2=args["filetype2"])
    if args.experiment == 5:
        experiment_training_logistic_function(architecture=args["architecture"], modeltype1=args["modeltype1"],
                                              modeltype2=args["modeltype2"], name1=args["seedname1"],
                                              name2=args["seedname2"], filetype1=args["filetype1"],
                                              filetype2=args["filetype2"], rf_level=args["rf_level"],
                                              layer_index=args["layer_index"])
