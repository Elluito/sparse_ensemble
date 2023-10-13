import torch
import time
import pickle
import argparse
import torchvision.transforms as transforms
from pathlib import Path
import torchvision
import torchessian
import loss_landscapes
import loss_landscapes.metrics as metrics
# libraries
import copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm

matplotlib.rcParams['figure.figsize'] = [18, 12]
# training hyperparameters
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 10 ** -2
BATCH_SIZE = 512
EPOCHS = 25
# contour plot resolution
STEPS = 40


def main(args):
    cifar10_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    cifar100_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # stats_to_use = cifar10_stats if args.dataset == "cifar10" else cifar100_stats
    stats_to_use = cifar10_stats
    current_directory = Path().cwd()
    data_path = "/datasets"
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        data_path = "/nobackup/sclaam/data"
    elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
        data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
    elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
        data_path = "./datasets"
    elif 'lla98-mtc03' == current_directory.owner() or "luisaam" in current_directory.__str__():
        data_path = "./datasets"

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

    trainset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=10, shuffle=False, num_workers=0)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    # ################################### model #############################
    from torchvision.models import resnet18, resnet50
    from alternate_models.resnet import ResNet50_rf, ResNet18_rf
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
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)

    if args.solution:
        net.load_state_dict(torch.load(args.solution)["net"])

    ###########################################################################
    # f1 = open("loss_data_fin_{}.pkl".format(args.name), "wb")
    x, y = next(iter(testloader))
    x, y = x.cuda(), y.cuda()
    net.cuda()
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    metric = metrics.Loss(criterion, x, y)
    #
    # print("Is going to begin the random plane data calculationn")
    # t0 = time.time()
    # loss_data_fin = loss_landscapes.random_plane(net, metric, 10, STEPS, normalization='filter',
    #                                              deepcopy_model=True)
    # t1 = time.time()
    # print("The calculation lasted {}s".format(t1 - t0))
    # pickle.dump(loss_data_fin, f1)
    # f1.close()

    print("Is going to begin the hessian spectrum calculation data calculation")
    t0 = time.time()
    torch.cuda.empty_cache()
    ################# With PyHessian     ###############################################
    # hessian_comp = hessian(net, criterion, data=(x, y), cuda=True)
    # density_eigen, density_weight = hessian_comp.density()
    # f2 = open("density_eigen_{}.pkl".format(args.name), "wb")
    # f3 = open("density_weight_{}.pkl".format(args.name), "wb")
    # pickle.dump(density_eigen, f2)
    # pickle.dump(density_weight, f3)
    # f2.close()
    # f3.close()
    m = 90
    l, w = torchessian.complete_mode.gauss_quadrature(
        net,
        criterion,
        testloader,
        m,
        buffer=m
    )
    f2 = open("l{}.pkl".format(args.name), "wb")
    f3 = open("w{}.pkl".format(args.name), "wb")
    pickle.dump(l, f2)
    pickle.dump(l, f3)
    f2.close()
    f3.close()
    # ################  With torchessian
    t1 = time.time()
    print("The calculation lasted {}s".format(t1 - t0))

    support = torch.linspace(-5, 20, 10000)
    fig, ax = plt.subplots(figsize=(15, 7))
    density = torchessian.F(support, l, w, m)
    ax.plot(support.numpy(), density.numpy(), color='b')
    ax.set_yscale('log')
    ax.set_yticks([10 ** (i - 7) for i in range(10)])
    ax.set_ylim(10 ** -6, 10 ** 3)
    # red_patch = mpatches.Patch(color='red', label='Without BatchNorm')
    # blue_patch = mpatches.Patch(color='blue', label='With BatchNorm')
    # plt.legend(handles=[red_patch, blue_patch])
    plt.title("Beginning of training ResNet50")
    plt.savefig("spectral_density{}.pdf".format(args.name))

    # l, w = torchessian.complete_mode.gauss_quadrature(
    #     net,
    #     criterion,
    #     testloader,
    #     m,
    #     buffer=m
    # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--type', default="normal", type=str, help='Type of implementation [normal,official]')
    parser.add_argument('--RF_level', default=0, type=int, help='Receptive field level')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', default="cifar10", type=str, help='Dataset to use [cifar10,cifar100]')
    parser.add_argument('--model', default="resnet50", type=str, help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--solution', '-s', default="",
                        help='solution to use')
    parser.add_argument('--name', '-n', default="no_name",
                        help='name of the loss files')

    args = parser.parse_args()
    main(args)
