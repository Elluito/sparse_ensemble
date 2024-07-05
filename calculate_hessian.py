import torch
import time
import pickle
import argparse
import torchvision.transforms as transforms
from pathlib import Path
import torchvision
import torchessian
from pyhessian import hessian
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
from alternate_models import *
from torch.utils.data import random_split

matplotlib.rcParams['figure.figsize'] = [18, 12]
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


# def model_():

def main(args):
    ##########################################
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

        cifar10_train, cifar10_val = random_split(trainset, [len(trainset) - 5000, 5000])

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
            cifar10_train, batch_size=128, shuffle=True, num_workers=args.num_workers)

        valloader = torch.utils.data.DataLoader(
            cifar10_val, batch_size=128, shuffle=True, num_workers=args.num_workers)

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

    ###########################################################################
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

    net = net.to(device)
    net.eval()

    ###########################################################################

    prefix = Path("{}/{}/{}/{}".format(args.folder, args.model, args.dataset, args.method))
    prefix.mkdir(parents=True, exist_ok=True)
    criterion = torch.nn.CrossEntropyLoss()

    ################### With PyHessian     ####################################

    if args.method == "pyhessian":
        one_batch_string = "one_batch" if args.batch_only else "{}_samples".format(args.eval_size)
        if args.batch_only:
            for x, y in valloader:
                break
            x, y = x.cuda(), y.cuda()
            hessian_comp = hessian(net, criterion, data=(x, y), cuda=True)

            t0 = time.time()
            density_eigen, density_weight = hessian_comp.density()
            f3 = open("{}/pyhessian_density_eigen_{}_{}_lvl_{}.pkl".format(prefix, one_batch_string, args.name,
                                                                           args.RF_level), "wb")
            f4 = open("{}/pyhessian_density_weight_{}_{}_lvl_{}.pkl".format(prefix, one_batch_string, args.name,
                                                                            args.RF_level), "wb")
            pickle.dump(density_eigen, f3)
            pickle.dump(density_weight, f4)
            f3.close()
            f4.close()

            t1 = time.time()

            print("The density calculation lasted {}s".format(t1 - t0))

            t0 = time.time()
            trace = hessian_comp.trace()
            f5 = open("{}/pyhessian_trace_{}_{}_lvl_{}.pkl".format(prefix, one_batch_string, args.name, args.RF_level),
                      "wb")
            pickle.dump(trace, f5)
            f5.close()
            t1 = time.time()

            print("The trace calculation lasted {}s".format(t1 - t0))

            t0 = time.time()
            top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(top_n=args.n_eigenvalues)
            f1 = open(
                "{}/pyhessian_eigenvalues_top_{}_{}_{}_lvl_{}.pkl".format(prefix, args.n_eigenvalues, one_batch_string,
                                                                          args.name, args.RF_level),
                "wb")
            pickle.dump(top_eigenvalues, f1)
            f1.close()
            f1 = open(
                "{}/pyhessian_eigenvectors_top_{}_{}_{}_lvl_{}.pkl".format(prefix, args.n_eigenvalues, one_batch_string,
                                                                           args.name, args.RF_level),
                "wb")
            pickle.dump(top_eigenvectors, f1)
            f1.close()

            t1 = time.time()
            print("The calculation of top eigenvalues and eigenvectors lasted {}s".format(t1 - t0))
        else:
            hessian_comp = hessian(net, criterion, dataloader=valloader, cuda=True)
            t0 = time.time()
            density_eigen, density_weight = hessian_comp.density(iter=args.n_eigenvalues)
            f3 = open("{}/pyhessian_density_eigen_{}_{}_lvl_{}.pkl".format(prefix, one_batch_string, args.name,
                                                                           args.RF_level), "wb")
            f4 = open("{}/pyhessian_density_weight_{}_{}_lvl_{}.pkl".format(prefix, one_batch_string, args.name,
                                                                            args.RF_level), "wb")
            pickle.dump(density_eigen, f3)
            pickle.dump(density_weight, f4)
            f3.close()
            f4.close()

            t1 = time.time()

            print("The density calculation lasted {}s".format(t1 - t0))

            t0 = time.time()
            trace = hessian_comp.trace(maxIter=200)
            f5 = open("{}/pyhessian_trace_{}_{}_lvl_{}.pkl".format(prefix, one_batch_string, args.name, args.RF_level),
                      "wb")
            pickle.dump(trace, f5)
            f5.close()
            t1 = time.time()
            print("The trace calculation lasted {}s".format(t1 - t0))

            t0 = time.time()
            top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(top_n=args.n_eigenvalues)
            f1 = open(
                "{}/pyhessian_eigenvalues_top_{}_{}_{}_lvl_{}.pkl".format(prefix, args.n_eigenvalues, one_batch_string,
                                                                          args.name, args.RF_level),
                "wb")
            pickle.dump(top_eigenvalues, f1)
            # f1.close()
            # f1 = open(
            #     "{}/pyhessian_eigenvectors_top_{}_{}_{}_lvl_{}.pkl".format(prefix, args.n_eigenvalues, one_batch_string,
            #                                                                args.name, args.RF_level),
            #     "wb")
            # pickle.dump(top_eigenvectors, f1)
            # f1.close()

            t1 = time.time()
            print("The calculation of top eigenvalues and eigenvectors lasted {}s".format(t1 - t0))

    # # ################  With torchessian

    if args.method == "torchessian":
        test_data = valloader
        if args.batch_only:
            for x, y in valloader:
                break
            test_data = [(x, y)]
        t0 = time.time()

        m = args.n_eigenvalues
        l, w = torchessian.complete_mode.gauss_quadrature(
            net,
            criterion,
            test_data,
            m,
            buffer=args.n_buffer
        )
        f2 = open("{}/l_{}_lvl_{}.pkl".format(prefix, args.name, args.RF_level), "wb")
        f3 = open("{}/w_{}_lvl_{}.pkl".format(prefix, args.name, args.RF_level), "wb")
        pickle.dump(l, f2)
        pickle.dump(w, f3)
        f2.close()
        f3.close()

        t1 = time.time()
        print("The calculation lasted {}s".format(t1 - t0))

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

    parser.add_argument('--distributed', '-d', default="", action="store_true",
                        help='Either use distributed computation for claculating the hessian')
    parser.add_argument('--batch_only', '-bo', default="", action="store_true",
                        help='compute the hessian spectra only on one batch')

    parser.add_argument('--name', '-n', default="no_name", help='name of the loss files, usually the seed name')
    parser.add_argument('--method', '-m', default="torchessian", help='Which method to calculate the hessian')
    parser.add_argument('--eval_set', '-es', default="val", help='On which set to performa the calculations')

    parser.add_argument('--folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where the output of the algorithm is going to be saved')
    parser.add_argument('--data_folder', default="/nobackup/sclaam/data", type=str,
                        help='Location of the dataset', required=True)
    parser.add_argument('--width', default=1, type=int, help='Width of the model')
    parser.add_argument('--eval_size', default=1000, type=int, help='How many images to use in the calculation')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size for loading data')
    parser.add_argument('--n_eigenvalues', default=90, type=int, help='Number of largest eigenvalues')
    parser.add_argument('--n_buffer', default=90, type=int, help='Batch Size for training')
    parser.add_argument('--input_resolution', default=224, type=int,
                        help='Input Resolution for Small ImageNet')
    parser.add_argument('--ffcv', action='store_true', help='Use FFCV loaders')

    parser.add_argument('--ffcv_train',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv",
                        type=str, help='FFCV train dataset')

    parser.add_argument('--ffcv_val',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv",
                        type=str, help='FFCV val dataset')

    args = parser.parse_args()

    main(args)
