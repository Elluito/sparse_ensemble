import os
import time
import pickle
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
from confidence_utils import *

def test(net, testloader=None, verbose=0, name="ckpt", save_folder="./checkpoint", args=None):
    # global best_acc, testloader, device, criterion
    net.eval()
    net = net.to(device)
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if batch_idx == 10:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            print("Predicted:{}".format(predicted.cpu().numpy()))
            print("Targets:{}".format(targets.cpu().numpy()))

    # Save checkpoint.
    acc = 100. * correct / total
    return acc


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
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": exclude_layers,
         "data_path": args.data_folder,
         "input_resolution": args.input_resolution

         })

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

    dense_accuracy_list = []
    pruned_accuracy_list = []
    files_names = []
    dense_ece_list = []
    pruned_ece_list = []
    search_string = "{}/{}_normal_{}_*_level_{}*{}*test_acc_*.pth".format(args.folder, args.model, args.dataset,
                                                                          args.RF_level, args.name)

    full_output_folder_name = "{}/{}/{}/{}".format(args.output_folder, args.model, args.dataset,
                                                   args.RF_level)
    output_directory = Path(full_output_folder_name)

    output_directory.mkdir(parents=True, exist_ok=True)

    things = list(glob.glob(search_string))

    # if len(things) < 2:
    #     search_string = "{}/{}_normal_{}_*_level_{}.pth".format(args.folder, args.model, args.dataset, args.RF_level)

    print("Glob text:{}".format(
        "{}/{}_normal_{}_*_level_{}*{}*test_acc_*.pth".format(args.folder, args.model, args.dataset, args.RF_level,
                                                              args.name)))
    print(things)
    saved_already = False
    for i, name in enumerate(
            glob.glob(search_string)):

        print(name)

        print("Device: {}".format(device))

        state_dict_raw = torch.load(name, map_location=device)

        dense_accuracy_list.append(state_dict_raw["acc"])
        net.load_state_dict(state_dict_raw["net"])

        print("Dense accuracy:{}".format(state_dict_raw["acc"]))
        print("Calculated Dense accuracy:{}".format(test(net, testloader=testloader)))

        dense_accuracies, dense_confidences, dense_max_prob_correct, dense_max_prob_incorrect, dense_topk_prob_correct, dense_topk_prob_incorrect, dense_topk_prob_correct_index, dense_topk_prob_incorrect_index, correct_index = get_correctness_dataloader(
            net, testloader, "cpu", args.topk)

        DENSE_ECE = calc_ece(dense_confidences, dense_accuracies)

        dense_ece_list.append(DENSE_ECE.item())

        prune_function(net, cfg)

        remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)

        pruned_accuracies, pruned_confidences, pruned_max_prob_correct, pruned_max_prob_incorrect, pruned_topk_prob_correct, pruned_topk_prob_incorrect, pruned_topk_prob_correct_index, pruned_topk_prob_incorrect_index, _ = get_correctness_dataloader(
            net, testloader, "cpu", args.topk)

        PRUNED_ECE = calc_ece(pruned_confidences, pruned_accuracies)

        pruned_ece_list.append(PRUNED_ECE.item())

        t0 = time.time()
        if args.ffcv:
            pruned_accuracy = test_ffcv(net, testloader=testloader, verbose=0)
        else:
            pruned_accuracy = test(net, testloader=testloader)

        t1 = time.time()

        print("Pruned accuracy:{}".format(pruned_accuracy))

        print("Time for inference: {}".format(t1 - t0))

        pruned_accuracy_list.append(pruned_accuracy)
        # weight_names, weights = zip(*get_layer_dict(net))
        #
        # zero_number = lambda w: (torch.count_nonzero(w == 0) / w.nelement()).cpu().numpy()
        #
        # pruning_rates_per_layer = list(map(zero_number, weights))

        seed_from_file1 = re.findall("_[0-9]_", name)

        print(seed_from_file1)

        seed_from_file2 = re.findall("_[0-9]_[0-9]_", name)

        print(seed_from_file2)

        seed_from_file3 = re.findall("\.[0-9]_", name)

        print(seed_from_file3)

        if seed_from_file3:

            seed_from_file = seed_from_file3[0].replace(".", "_")

        elif seed_from_file2:

            seed_from_file = seed_from_file2[0].split("_")[2]

        elif seed_from_file1:

            seed_from_file = seed_from_file1[0].replace("_", "")
        else:
            seed_from_file = i

        # df2 = pd.DataFrame({"layer_names": weight_names, "pr": pruning_rates_per_layer})
        # print("Seed from file {}".format(seed_from_file1))
        # df2.to_csv(
        #     "{}_level_{}_seed_{}_{}_{}_pruning_rates_global_pr_{}.csv".format(args.model, args.RF_level, seed_from_file,
        #                                                                       args.dataset, args.name,
        #                                                                       args.pruning_rate),
        #     index=False)
        if not saved_already:
            # Pruned max prob correct, incorrect and top 5 prob for incorrect and correct for the pruned
            with open("{}/seed_{}_data/pruned_{}/max_prob_correct.pkl".format(output_directory, i, args.pruning_rate),
                      "wb") as f:
                pickle.dump(pruned_max_prob_correct, f)
            with open("{}/seed_{}_data/pruned_{}/max_prob_incorrect.pkl".format(output_directory, i, args.pruning_rate),
                      "wb") as f:
                pickle.dump(pruned_max_prob_incorrect, f)
            with open("{}/seed_{}_data/pruned_{}/topk_correct.pkl".format(output_directory, i, args.pruning_rate),
                      "wb") as f:
                pickle.dump(pruned_topk_prob_correct, f)
            with open("{}/seed_{}_data/pruned_{}/topk_incorrect.pkl".format(output_directory, i, args.pruning_rate),
                      "wb") as f:
                pickle.dump(pruned_topk_prob_incorrect, f)
            with open("{}/seed_{}_data/pruned_{}/topk_incorrect_index.pkl".format(output_directory, i, args.pruning_rate),
                      "wb") as f:
                pickle.dump(pruned_topk_prob_incorrect_index, f)

            # Pruned max prob correct, incorrect and top 5 prob for incorrect and correct for the  Dense
            with open("{}/seed_{}_data/dense/max_prob_correct.pkl".format(output_directory, i),
                      "wb") as f:
                pickle.dump(dense_max_prob_correct, f)
            with open("{}/seed_{}_data/dense/max_prob_incorrect.pkl".format(output_directory, i),
                      "wb") as f:
                pickle.dump(dense_max_prob_incorrect, f)
            with open("{}/seed_{}_data/dense/topk_correct.pkl".format(output_directory, i),
                      "wb") as f:
                pickle.dump(dense_topk_prob_correct, f)
            with open("{}/seed_{}_data/dense/topk_incorrect.pkl".format(output_directory, i),
                      "wb") as f:
                pickle.dump(dense_topk_prob_incorrect, f)
            with open("{}/seed_{}_data/dense/accuracies.pkl".format(output_directory, i), "wb") as f:
                pickle.dump(dense_accuracies, f)
            with open("{}/seed_{}_data/dense/topk_incorrect_index.pkl".format(output_directory, i),
                      "wb") as f:
                pickle.dump(dense_topk_prob_incorrect_index, f)
            with open("{}/seed_{}_data/dense/correct_label.pkl".format(output_directory, i),
                      "wb") as f:
                pickle.dump(correct_index, f)

            saved_already = True

        print("Done")
        file_name = os.path.basename(name)
        print(file_name)
        files_names.append(file_name)

    df = pd.DataFrame({"Name": files_names,
                       "Dense Accuracy": dense_accuracy_list,
                       "Pruned Accuracy": pruned_accuracy_list,
                       "Dense ECE": dense_ece_list,
                       "Pruned ECE": pruned_ece_list
                       })
    df.to_csv(
        "{}/accuracy_and_confidence_summary_{}_pr_{}.csv".format(output_directory,
                                                                 args.name, args.pruning_rate),
        index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Confidence statistics for models ')

    parser.add_argument('--experiment', default=1, type=int, help='Experiment to perform')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--type', default="normal", type=str, help='Type of implementation [normal,official]')
    parser.add_argument('--RF_level', default="4", type=str, help='Receptive field level')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', default="cifar10", type=str, help='Dataset to use [cifar10,tiny_imagenet616gg]')
    parser.add_argument('--model', default="resnet18", type=str, help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where saved models are')
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
    parser.add_argument('--topk', default=10, type=int,
                        help='How much top probabilities to take')
    parser.add_argument('--ffcv', action='store_true', help='Use FFCV loaders')

    parser.add_argument('--ffcv_train',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv",
                        type=str, help='FFCV train dataset')

    parser.add_argument('--ffcv_val',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv",
                        type=str, help='FFCV val dataset')
    parser.add_argument("--output_folder", default="~/sparse_ensemble/confidence_and_RF", type=str,
                        help='Location where saved models are')

    args = parser.parse_args()

    try:

        args.RF_level = int(args.RF_level)

    except Exception as e:

        pass

    if args.experiment == 1:
        print("Experiment 1")
        print(args)
        main(args)
