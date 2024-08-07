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

matplotlib.use('tkAgg')
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


def whole_dataset_loss(model, dataloader, no_use_y):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.data.item()
    return test_loss


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
        trainset, batch_size=1000, shuffle=True, num_workers=0)

    indices = torch.arange(0, 10000)
    smaller_trainset = torch.utils.data.Subset(trainset, indices)
    trainloader_hessian = torch.utils.data.DataLoader(
        smaller_trainset, batch_size=10, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test)

    testloader1 = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=0)

    testloader2 = torch.utils.data.DataLoader(
        testset, batch_size=1000, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # ################################### model #############################
    from torchvision.models import resnet18, resnet50
    from alternate_models.resnet import ResNet50_rf, ResNet18_rf
    from alternate_models.vgg import VGG_RF

    print("Solution")
    print(args.solution)
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

    if args.model == "vgg19":
        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, rf_level=args.RF_level)

    if args.solution:
        temp_dict = torch.load(args.solution, map_location=torch.device('cpu'))["net"]
        if args.type == "normal" and args.RF_level != 0:
            net.load_state_dict(temp_dict)
            print("Loaded solution!")
        else:
            real_dict = {}
            for k, item in temp_dict.items():
                if k.startswith('module'):
                    new_key = k.replace("module.", "")
                    real_dict[new_key] = item
            net.load_state_dict(real_dict)
            print("Loaded solution!")

    ###########################################################################
    from sparse_ensemble_utils import test
    # training_test = test(net, use_cuda=False, testloader=trainloader_hessian, verbose=1)
    # print("Accuracy of 10k samples of training set {}".format(training_test))
    # return
    prefix = Path("/nobackup/sclaam/smoothness/{}/{}".format(args.model,args.dataset))
    prefix.mkdir(parents=True, exist_ok=True)
    # f1 = open("{}/loss_data_fin_{}.pkl".format(prefix, args.name), "wb")
    # # f1 = open("loss_data_fin_train{}.pkl".format(args.name), "wb")
    # x, y = next(iter(trainloader))
    # x, y = x.cuda(), y.cuda()
    # net.cuda()
    # net.eval()
    # print(len(x))
    criterion = torch.nn.CrossEntropyLoss()
    metric = metrics.sl_metrics.BatchedLoss(criterion, trainloader_hessian)
    #
    # #
    # print("Is going to begin the random plane data calculation")
    # t0 = time.time()
    loss_data_fin = loss_landscapes.random_plane(net, metric, 0.15, STEPS, normalization='filter',
                                                 deepcopy_model=True)
    # t1 = time.time()
    # print("The calculation lasted {}s".format(t1 - t0))
    #
    # print(loss_data_fin)
    # pickle.dump(loss_data_fin, f1)
    # f1.close()

    #   Plotting ########################################

    # from smoothness_plotting import plot_3d, countour_plot
    # f1 = open("loss_data_fin_train{}.pkl".format(args.name), "rb")
    # loss_data_fin = pickle.load(f1)
    # f1.close()
    # countour_plot(loss_data_fin)
    # plot_3d(loss_data_fin, "{}_train".format(args.name), "Pytorch seed 1 trainset", save=False)

    # print("Is going to begin the hessian spectrum calculation data calculation")
    # t0 = time.time()
    # torch.cuda.empty_cache()
    # ################# With PyHessian     ###############################################
    # # hessian_comp = hessian(net, criterion, data=(x, y), cuda=True)
    # # density_eigen, density_weight = hessian_comp.density()
    # # f2 = open("density_eigen_{}.pkl".format(args.name), "wb")
    # # f3 = open("density_weight_{}.pkl".format(args.name), "wb")
    # # pickle.dump(density_eigen, f2)
    # # pickle.dump(density_weight, f3)
    # # f2.close()
    # # f3.close()

    # # ################  With torchessian

    m = 90
    l, w = torchessian.complete_mode.gauss_quadrature(
        net,
        criterion,
        trainloader_hessian,
        m,
        buffer=m
    )
    f2 = open("{}/l_{}.pkl".format(prefix, args.name), "wb")
    f3 = open("{}/w_{}.pkl".format(prefix, args.name), "wb")
    pickle.dump(l, f2)
    pickle.dump(w, f3)
    f2.close()
    f3.close()

    # t1 = time.time()
    # print("The calculation lasted {}s".format(t1 - t0))
    #
    # support = torch.linspace(-5, 20, 10000)
    # fig, ax = plt.subplots(figsize=(15, 7))
    # density = torchessian.F(support, l, w, m)
    # ax.plot(support.numpy(), density.numpy(), color='b')
    # ax.set_yscale('log')
    # ax.set_yticks([10 ** (i - 7) for i in range(10)])
    # ax.set_ylim(10 ** -6, 10 ** 3)
    # # red_patch = mpatches.Patch(color='red', label='Without BatchNorm')
    # # blue_patch = mpatches.Patch(color='blue', label='With BatchNorm')
    # # plt.legend(handles=[red_patch, blue_patch])
    # plt.title("Beginning of training ResNet50")
    # plt.savefig("{}spectral_density{}.pdf".format(prefix, args.name))

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
    parser.add_argument('--type', default="pytorch", type=str, help='Type of implementation [normal,pytorch]')
    parser.add_argument('--RF_level', default=0, type=int, help='Receptive field level')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', default="cifar10", type=str, help='Dataset to use [cifar10,cifar100]')
    parser.add_argument('--model', default="resnet50", type=str, help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--solution', '-s',
                        default="foreing_trained_models/cifar10/resnet50_official_cifar10_seed_1_test_acc_90.31.pth",
                        help='solution to use')
    parser.add_argument('--name', '-n', default="no_name", help='name of the loss files, usually the seed name')

    args = parser.parse_args()
    main(args)
