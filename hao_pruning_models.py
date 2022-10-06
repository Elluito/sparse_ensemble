import sys
import pickle
from typing import List
import pandas as pd
import datetime as date
import optuna
# sys.path.append('csgmcmc')
from csgmcmc.models import *
import omegaconf
import copy
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import hydra
import torchvision
import torchvision.transforms as transforms
import scipy
import os
import argparse
import scipy.optimize as optimize
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.utils.prune as prune
import platform
import matplotlib
from functools import partial

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FormatStrFormatter
import sklearn as sk
from sklearn.manifold import MDS, TSNE
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time


def load_model(net, path):
    net.load_state_dict(torch.load(path))


def weigths_to_prune(model):
    modules = []
    for m in model.modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            modules.append((m, "weight"))
    return modules


def remove_reparametrization(model, name_module=""):
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            if name_module == "":
                prune.remove(m, "weight")
            if name == name_module:
                prune.remove(m, "weight")
                break


def test(net, use_cuda, testloader, one_batch=False):
    if use_cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % 100 == 0:
                print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct.item() / total, correct, total))
            if one_batch:
                return 100. * correct.item() / total
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss / len(testloader), correct, total,
        100. * correct.item() / total))
    return 100. * correct.item() / total


def get_models_pruned(cfg):
    data_path = cfg.data_path
    use_cuda = torch.cuda.is_available()

    # This comes from the """from csgmcmc.models import *""" import of the models.
    # Acording to me they don't have any particular detail specific to bayesian networks.
    if cfg.architecture == "resnet18":
        net = ResNet18()

    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=1)
    load_model(net, "trained_models/cifar10/cifar_csghmc_5.pt")
    prunings_percentages = [0.9, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
                            0.2, 0.1]

    for pr in prunings_percentages:
        current_model = copy.deepcopy(net)
        weights_to_prune = weigths_to_prune(current_model)
        prune.global_unstructured(
            weights_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pr
        )
        remove_reparametrization(current_model)
        print(f"Performance of model pruned {pr}")
        pruned_original_performance = test(current_model, use_cuda, testloader)
        torch.save(current_model, f"pruned_models/pruned_{cfg.architecture}_pr_{pr}.pt")


if __name__ == '__main__':
    cfg = omegaconf.DictConfig({
        "architecture": "resnet18",
        "batch_size": 100,
        # here is were the cifar10 dataset is going to be stored
        "data_path": "/nobackup/sclaam/data" if platform.system() != "Windows" else "."
    })
    get_models_pruned(cfg)

