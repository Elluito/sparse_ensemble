import argparse
import glob
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import omegaconf

from main import prune_function, remove_reparametrization
from train_CIFAR10 import get_model
from saturation_utils import calculate_train_eval_saturation_solution

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_saturation_models(args):

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
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": args.pruning_rate,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
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
        print('==> Preparing data..')
        data_path = args.data_folder
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
                 "num_workers": args.num_workers, "batch_size": batch_size,"input_resolution":64})
        if args.dataset == "small_imagenet":
            if args.ffcv:
                from ffcv_loaders import make_ffcv_small_imagenet_dataloaders
                trainloader, valloader, testloader = make_ffcv_small_imagenet_dataloaders(args.ffcv_train,
                                                                                          args.ffcv_val,
                                                                                          batch_size, args.num_workers,
                                                                                          resolution=args.input_resolution,
                                                                                          valsize=0)
            else:
                from test_imagenet import load_small_imagenet
                trainloader, valloader, testloader = load_small_imagenet(
                    {"traindir": data_path + "/small_imagenet/train", "valdir": data_path + "/small_imagenet/val",
                     "num_workers": args.num_workers, "batch_size": batch_size, "resolution": args.input_resolution,
                     "resize": args.resize})

    net = get_model(args)

    search_string = "{}/{}_normal_{}_*_level_{}_*{}*test_acc_*.pth".format(args.folder, args.model, args.dataset,
                                                                           args.RF_level, args.name)
    things = list(glob.glob(search_string))

    print("Glob text:{}".format(
        "{}/{}_normal_{}_*_level_{}_*{}*test_acc_*.pth".format(args.folder, args.model, args.dataset, args.RF_level,
                                                               args.name)))
    print(things)
    sufix_name = "{}_{}_{}_{}_pr_{}_saturation_trained".format(args.model,
                                                       args.RF_level, args.dataset,
                                                       args.name, args.pruning_rate)
    for i, name in enumerate(
            glob.glob(search_string)):
        state_dict_raw = torch.load(name, map_location=device)

        net.load_state_dict(state_dict_raw["net"])
        if args.pruning_rate != 0:
            prune_function(net, cfg)
            remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)
        calculate_train_eval_saturation_solution(net, trainloader, testloader, args.save_folder, sufix_name, i, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saturation calculation')

    parser.add_argument('--type', default="normal", type=str, help='Type of implementation [normal,official]')
    parser.add_argument('--RF_level', default="4", type=str, help='Receptive field level')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', default="cifar10", type=str,
                        help='Dataset to use [cifar10,tiny_imagenet]')
    parser.add_argument('--model', default="resnet18", type=str,
                        help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where saved models are')
    parser.add_argument('--save_folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Output folder of the saturation results')
    parser.add_argument('--data_folder', default="/mnt/scratch/sclaam/data2", type=str,
                        help='Location of datasets', required=True)
    parser.add_argument('--resize', default=0, type=int,
                        help='Either resize the image to 32x32 and then back to input resolution')
    parser.add_argument('--name', default="", type=str, help='Name of the file', required=False)
    parser.add_argument('--pruning_rate', default=0.9, type=float, help='Pruning rate')
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

    try:
        args.RF_level = int(args.RF_level)
    except Exception as e:
        pass

    calculate_saturation_models(args)
