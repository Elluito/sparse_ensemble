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
import glob

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FormatStrFormatter
import sklearn as sk
from sklearn.manifold import MDS, TSNE
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from torch.utils.data import DataLoader, random_split, Dataset
import logging
import torchvision as tv
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from itertools import chain, combinations


# function taken from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# #sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, convert_to_int=True, percentage=False, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if convert_to_int:
                kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
                text = im.axes.text(j, i, valfmt(int(data[i, j]), None), **kw)
                texts.append(text)
            else:
                if percentage:
                    kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
                    text = im.axes.text(j, i, valfmt(format_percentages(data[i, j]), None), **kw)
                    texts.append(text)
                else:
                    kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
                    text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                    texts.append(text)

    return texts


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1)))


def strip_prefix(net_state_dict: dict):
    new_dict = {}
    for k, v in net_state_dict.items():
        new_key = k.replace("module.", "")
        new_dict[new_key] = v
    return new_dict


def load_model(net, path):
    state_dict = torch.load(path)
    if "net" in state_dict.keys():
        net.load_state_dict(strip_prefix(state_dict["net"]))
    else:
        net.load_state_dict(torch.load(path))


def get_layer_dict(model):
    iter_1 = model.named_modules()
    layer_dict = []

    for name, m in iter_1:
        with torch.no_grad():
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not \
                    isinstance(m, nn.BatchNorm3d):
                layer_dict.append((name, torch.flatten(m.weight.data).cpu().detach()))
                # if "shortcut" in name:
                #     print(f"{name}:{m.weight.data}")
                # if not isinstance(m, nn.Conv2d):
                #     print(f"{name}:{m.weight.data}")

    return layer_dict


def save_onnx(cfg):
    data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
                                                                               "University of Leeds\PhD\Datasets\CIFAR10"
    use_cuda = torch.cuda.is_available()
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)
    batch = next(iter(testloader))
    input_names = ['Image']
    output_names = ['y_hat']
    torch.onnx.export(net, batch[0], f'onnx_models/onnx_models_{cfg.architecture}.onnx', input_names=input_names,
                      output_names=output_names)


def models_inspection(population):
    total_params = count_parameters(population[0])
    layers_of_models = [get_layer_dict(mod) for mod in population]
    sparsity = lambda w: torch.count_nonzero(w == 0) / w.nelement()
    zero_number = lambda w: torch.count_nonzero(w == 0) / total_params
    tsne = []
    mds = []
    MDS_embeddings = MDS(n_components=2)
    tsne_embeddings = TSNE(n_components=2)
    plt.figure(figsize=(10, 15), dpi=80)
    points = defaultdict(list)

    for i, layer in enumerate(layers_of_models):
        names, weigths = zip(*layer)
        sparsities = list(map(zero_number, weigths))
        plt.bar(x=np.arange(0, len(sparsities), 1), height=sparsities, tick_label=names)
        plt.xticks(rotation=-45)
        plt.ylabel("Percentage of total parameters", fontsize=20)
        plt.title("Number of zero weights", fontsize=20)
        plt.savefig("data/figures/bar_plot_model_{}.png".format(i))
        plt.close()
        for element in layer:
            # MDS.append("Model {} {}".format(i,element[0]),MDS_embeddings.fit_transform(element[1]))
            # embedding = MDS_embeddings.fit_transform(element[1].reshape(1,-1))
            label = "M:{}L:{}".format(i, element[0])
            points[element[0]].append((label, element[1]))

    fig, ax = plt.subplots()
    for layer_name, plotting_stuf in points.items():
        point_names, tensors = zip(*plotting_stuf)
        tensors = torch.stack(tensors)
        embeddings = tsne_embeddings.fit_transform(tensors)
        ax.scatter(embeddings[0], embeddings[1], label=layer_name)
        # ax.annotate(point_names[k], (emb[0], emb[1]))
    plt.ylabel("D1", fontsize=20)
    plt.xlabel("D2", fontsize=20)
    plt.title("T-SNE", fontsize=20)
    ax.legend()
    # plt.show()
    plt.savefig("data/figures/layer_projection_tsne.png")
    plt.close()

    fig, ax = plt.subplots()
    list_of_names = list(points.keys())
    for layer_name, plotting_stuf in points.items():
        point_names, tensors = zip(*plotting_stuf)
        tensors = torch.stack(tensors)
        embeddings = MDS_embeddings.fit_transform(tensors)
        ax.scatter(embeddings[:, 0], embeddings[:, 1], label=layer_name)
        # ax.annotate(point_names[k], (emb[0], emb[1]))
    plt.ylabel("D1", fontsize=20)
    plt.xlabel("D2", fontsize=20)
    plt.title("MDS", fontsize=20)
    ax.legend()
    # plt.show()
    plt.savefig("data/figures/layer_projection_MDS.png")
    plt.close()


def plot_pareto(df, title: str):
    # define aesthetics for plot
    color1 = 'steelblue'
    color2 = 'red'
    line_size = 4
    fig, ax = plt.subplots()
    ax.bar(df.index, df['count'], color=color1)

    # add cumulative percentage line to plot
    ax2 = ax.twinx()
    ax2.plot(df.index, df['cumperc'], color=color2, marker="D", ms=line_size)
    ax2.yaxis.set_major_formatter(PercentFormatter())

    # specify axis colors
    ax.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)

    ax.tick_params(axis="x", labelrotation=90)
    ax2.tick_params(axis="x", labelrotation=90)
    plt.title(title, fontsize=20)
    # display Pareto chart
    plt.show()


def weight_inspection(cfg, cutoff):
    """
    This function inspects statistics about the weights of the pruned models
    :return:
    """
    performances = np.load("data/population_data/performances_{}.npy".format(cfg.noise))
    with open("data/population_data/population_models_{}.pkl".format(cfg.noise), "rb") as f:
        pop = pickle.load(f)
    sorted_indx = np.argsort(performances)
    sorted_models = [pop[i] for i in sorted_indx]

    above_cutoff_index = np.where(performances[sorted_indx] >= cutoff)[0]
    underneath_cutoff_index = np.where(performances[sorted_indx] < cutoff)[0]
    # measure_overlap(sorted_models[0],sorted_models[1])

    # models_inspection(sorted_models[:5])
    total_params = count_parameters(pop[0])
    zero_number = lambda w: torch.count_nonzero(w == 0)
    number_param = lambda w: w.nelement()
    layer = get_layer_dict(pop[0])
    names, weights = zip(*layer)
    elements = list(map(number_param, weights))

    sparsities = list(map(zero_number, weights))
    weights = list(map(lambda w: w.numpy(), weights))
    sparsities = list(map(lambda w: w.numpy(), sparsities))
    # elements =list(map(lambda w: w.numpy(),elements))
    df = pd.DataFrame({'count': elements})
    df.index = names
    df = df.sort_values(by='count', ascending=False)
    df['cumperc'] = df['count'].cumsum() / df['count'].sum() * 100
    plot_pareto(df, "Number of 0 parameters")
    # Go trough all the above_cutoff_index


def add_geometric_gaussian_noise_to_weights(m, sigma=0.2):
    with torch.no_grad():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            m.weight.multiply_(torch.normal(mean=torch.ones_like(m.weight), std=sigma).to(m.weight.device))


def add_gaussian_noise_to_weights(m, sigma=0.01):
    with torch.no_grad():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            m.weight.add_(torch.normal(mean=torch.zeros_like(m.weight), std=sigma).to(m.weight.device))


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
    net.cpu()
    return 100. * correct.item() / total


def weigths_to_prune(model):
    modules = []
    for m in model.modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            modules.append((m, "weight"))
    return modules


def get_weights_of_layer_name(model, layer_name):
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            if name == layer_name:
                return (m, "weight")


def remove_reparametrization(model, name_module=""):
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            if name_module == "":
                prune.remove(m, "weight")
            if name == name_module:
                prune.remove(m, "weight")
                break


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_zero_parameters(model):
    return sum((p == 0).sum() for p in model.parameters() if p.requires_grad)


def get_proba_function(C, N):
    """
    :param C: pruning rate
    :param N:  number of parameters
    :return: function
    """

    def loss(k):
        return ((1 - np.exp(-k * N)) / k - C) ** 2

    result = optimize.minimize(loss, np.array([1]))
    print("K:{}".format(result["x"]))
    K = result["x"]

    def function(x):
        return np.exp(-K * x)

    return function


############################### 27 of september experiments ###################################################
def heatmap1_exp(cfg):
    # data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
    #                                                                            "University of Leeds\PhD\Datasets\CIFAR10"
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18":
        net = ResNet18()

    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    # for filename in glob.glob("trained_models/cifar10/*.pt"):

    load_model(net, cfg.solution)
    # model_index= filename[filename.index(".")-1]
    N = cfg.population
    number_populations = 3
    original_performance = test(net, use_cuda, testloader)
    pruning_percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pruning_percentages.reverse()
    sigmas = None
    if cfg.noise == "gaussian":
        sigmas = np.linspace(0.001, 0.01, 10)
    if cfg.noise == "geogaussian":
        sigmas = np.linspace(0.1, 0.7, 10)
    matrix_mean_decrease = np.zeros((len(sigmas), len(pruning_percentages)))
    matrix_ranking = np.zeros((len(sigmas), len(pruning_percentages)))
    deterministic_pruning_acc = []
    for i, pruning_percentage in enumerate(pruning_percentages):

        # Here I just prune once for the deterministic pruning

        pruned_original = copy.deepcopy(net)
        weights_to_prune = weigths_to_prune(pruned_original)
        prune.global_unstructured(
            weights_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_percentage
        )
        remove_reparametrization(pruned_original)
        print("performance of pruned original")
        pruned_original_performance = test(pruned_original, use_cuda, testloader)

        for j, sigma in enumerate(sigmas):
            performance_diff = []
            performances = [pruned_original_performance]
            ranks_pruned_model = []
            for _ in range(number_populations):

                for n in range(N):
                    current_model = copy.deepcopy(net)
                    if cfg.noise == "gaussian":

                        current_model.apply(partial(add_gaussian_noise_to_weights, sigma=sigma))
                    elif cfg.noise == "geogaussian":

                        current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=sigma))

                    prune.global_unstructured(weigths_to_prune(current_model),
                                              pruning_method=prune.L1Unstructured,
                                              amount=pruning_percentage)

                    remove_reparametrization(current_model)
                    print("Performance for model {}".format(n))
                    current_model_acc = test(current_model, use_cuda, testloader)
                    performances.append(current_model_acc)
                    performance_diff.append((original_performance - current_model_acc) / original_performance)

                sorted_performances = sorted(performances)
                sorted_performances.reverse()
                rank_original_model = sorted_performances.index(pruned_original_performance)
                ranks_pruned_model.append(rank_original_model)

            matrix_mean_decrease[j, i] = np.mean(performance_diff)
            matrix_ranking[j, i] = np.mean(ranks_pruned_model)
            print("Matrix Mean Decrease")
            print(np.matrix(matrix_mean_decrease))
            print("Matrix Ranking")
            print(np.matrix(matrix_ranking))
            # matrix_mean_decrease[j, i] = i+j
        # matrix_ranking[j, i] = i-j

    df_ranking = pd.DataFrame(matrix_ranking, columns=pruning_percentages)
    df_mean_decrease = pd.DataFrame(matrix_mean_decrease, columns=pruning_percentages)
    df_ranking.index = sigmas
    df_mean_decrease.index = sigmas
    df_ranking.to_csv(f"data/ranking_matrix_{cfg.noise}_traditional_train.csv")
    df_mean_decrease.to_csv(f"data/mean_decrease_matrix_{cfg.noise}_traditional_train.csv")
    ####################################  mean decrease heatmap #################################
    sigmas = list(map(format_sigmas, sigmas))
    fig, ax = plt.subplots(figsize=(5.1, 5.1))
    im = ax.imshow(matrix_mean_decrease.transpose(), aspect=0.5)
    plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
    plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
    # plt.yticks(ticks=pruning_percentages)
    # plt.xticks(ticks=sigmas)
    plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
    plt.ylabel("Pruning Rate", fontsize=12)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Mean decrease in performance', rotation=270, labelpad=8.5)
    # plt.gcf().set_size_inches(5.1, 5.1)
    texts = annotate_heatmap(im, valfmt="{x}")
    plt.title(f"Decrease in performance model", fontsize=15)
    plt.tight_layout()
    result = time.localtime(time.time())
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    plt.savefig(
        f"data/figures/meanDecrease_pruning_V_{cfg.noise}Noise_traditional_train_{result.tm_hour}-{result.tm_min}.pdf")
    plt.savefig(
        f"data/figures/meanDecrease_pruning_V_{cfg.noise}Noise_traditional_train_{result.tm_hour}-{result.tm_min}.png")
    plt.close()
    ####################################  ranking of original pruned model heatmap #################################
    ax = plt.subplot()
    im = ax.imshow(matrix_ranking.transpose(), aspect=0.5)
    ax.set_xticklabels(sigmas, minor=False)
    ax.set_yticklabels(pruning_percentages, minor=False)
    plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
    plt.ylabel("Pruning Rate", fontsize=12)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Rank', rotation=270, labelpad=8.5)
    # plt.gcf().set_size_inches(5.1, 5.1)
    plt.title(f"Rank of det model")
    texts = annotate_heatmap(im, valfmt="{x}")
    plt.tight_layout()
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    result = time.localtime(time.time())
    plt.savefig(
        f"data/figures/ranking_pruning_V_{cfg.noise}Noise_traditional_train_{result.tm_hour}-{result.tm_min}.pdf")
    plt.savefig(
        f"data/figures/ranking_pruning_V_{cfg.noise}Noise_traditional_train_{result.tm_hour}-{result.tm_min}.png")
    plt.close()


def copy_buffers(from_net: nn.Module, to_net: nn.Module):
    iter_1 = to_net.named_modules()

    for name, m in iter_1:
        with torch.no_grad():
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not \
                    isinstance(m, nn.BatchNorm3d):
                weight_mask = dict(dict(from_net.named_modules())[name].named_buffers())["weight_mask"]
                m.weight.data.mul_(weight_mask)


def heatmap2_mean_decrease_maskTransfer_exp(cfg):
    data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
                                                                               "University of Leeds\PhD\Datasets\CIFAR10"
    use_cuda = torch.cuda.is_available()
    net = None
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)
    load_model(net, "trained_models/cifar10/cifar_csghmc_5.pt")
    N = cfg.population
    pop = []
    performance = []
    number_of_populations = 3
    original_performance = test(net, use_cuda, testloader)
    pruning_percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pruning_percentages.reverse()
    sigmas = None
    if cfg.noise == "gaussian":
        sigmas = np.linspace(0.001, 0.01, 10)
    if cfg.noise == "geogaussian":
        sigmas = np.linspace(0.1, 0.7, 10)
    matrix_mean_decrease = np.zeros((len(sigmas), len(pruning_percentages)))
    matrix_ranking = np.zeros((len(sigmas), len(pruning_percentages)))
    deterministic_pruning_acc = []
    for i, pruning_percentage in enumerate(pruning_percentages):

        # Here I just prune once for the deterministic pruning

        pruned_original = copy.deepcopy(net)
        weights_to_prune = weigths_to_prune(pruned_original)
        prune.global_unstructured(
            weights_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_percentage
        )
        remove_reparametrization(pruned_original)
        print("performance of pruned original")
        pruned_original_performance = test(pruned_original, use_cuda, testloader)

        for j, sigma in enumerate(sigmas):
            # mean_index = []
            # for k in range(number_of_populations):
            pruned_original = copy.deepcopy(net)

            performance_diff = []
            # performances = [pruned_original_performance]
            # Loop over the  population
            for n in range(N):
                current_model = copy.deepcopy(net)
                if cfg.noise == "gaussian":

                    current_model.apply(partial(add_gaussian_noise_to_weights, sigma=sigma))
                elif cfg.noise == "geogaussian":

                    current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=sigma))

                prune.global_unstructured(
                    weigths_to_prune(current_model),
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_percentage)
                copy_buffers(from_net=current_model, to_net=pruned_original)

                # remove_reparametrization(current_model)

                print("Performance for model {}".format(n))

                current_model_acc = test(pruned_original, use_cuda, testloader)
                performance_diff.append((current_model_acc - pruned_original_performance) / pruned_original_performance)
                # current_model.cpu()
                # performances.append(current_model_acc)
            #

            # sorted_performances = sorted(performances)
            # sorted_performances.reverse()
            # rank_original_model = sorted_performances.index(pruned_original_performance)
            # mean_index.append(rank_original_model)
            matrix_mean_decrease[j, i] = np.mean(performance_diff)
            # matrix_ranking[j, i] = np.mean(mean_index)
            print("Matrix Mean Decrease")
            print(np.matrix(matrix_mean_decrease))
            # print("Matrix Ranking")
            # print(np.matrix(matrix_ranking))
        # matrix_mean_decrease[j, i] = i+j
        # matrix_ranking[j, i] = i-j

    df_mean_decrease = pd.DataFrame(matrix_mean_decrease, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_mean_decrease, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/buffer_transfer_meanDecrease_matrix_{cfg.noise}_pop_{cfg.population}.csv")
    # df_mean_decrease.to_csv(f"data/mean_decrease_matrix_{cfg.noise}.csv")
    ####################################  mean decrease heatmap #################################
    sigmas = list(map(format_sigmas, sigmas))
    ####################################  ranking of original pruned model heatmap #################################
    ax = plt.subplot()
    im = ax.imshow(matrix_mean_decrease.transpose(), aspect=0.5)
    # ax.set_xticklabels(sigmas, minor=False, rotation=-90)
    # ax.set_yticklabels(pruning_percentages, minor=False)
    plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
    plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
    plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
    plt.ylabel("Pruning Rate", fontsize=12)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Mean decrease', rotation=270, labelpad=8.5)
    # plt.gcf().set_size_inches(5.1, 5.1)
    plt.title("Mask transfer: {}".format(cfg.noise))
    texts = annotate_heatmap(im, valfmt="{x}")
    plt.tight_layout()
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    result = time.localtime(time.time())
    plt.savefig(f"data/figures/meanDecrease_bufferTransfer_pruning_V_{cfg.noise}Noise_{result.tm_hour}"
                f"-{result.tm_min}.png")
    plt.savefig(f"data/figures/meanDecrease_bufferTransfer_pruning_V_{cfg.noise}Noise_{result.tm_hour}"
                f"-{result.tm_min}.pdf")
    plt.close()


def heatmap2_mean_decrease_maskTransfer_exp_to_stochastic(cfg):
    data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
                                                                               "University of Leeds\PhD\Datasets\CIFAR10"
    use_cuda = torch.cuda.is_available()
    net = None
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                             num_workers=cfg.num_workers)
    load_model(net, "trained_models/cifar10/cifar_csghmc_5.pt")
    N = cfg.population
    pruning_percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # pruning_percentages.reverse()

    sigmas = None
    if cfg.noise == "gaussian":
        sigmas = np.linspace(0.001, 0.01, 10)
    if cfg.noise == "geogaussian":
        sigmas = np.linspace(0.1, 0.7, 10)
    matrix_diff = np.zeros((len(sigmas), len(pruning_percentages)))
    matrix_stochastic_acc = np.zeros((len(sigmas), len(pruning_percentages)))
    matrix_transfer_mask_acc = np.zeros((len(sigmas), len(pruning_percentages)))
    for i, pruning_percentage in enumerate(pruning_percentages):

        # Here I just prune once for the deterministic pruning

        # pruned_original = copy.deepcopy(net)
        # weights_to_prune = weigths_to_prune(pruned_original)
        # prune.global_unstructured(
        #     weights_to_prune,
        #     pruning_method=prune.L1Unstructured,
        #     amount=pruning_percentage
        # )
        # remove_reparametrization(pruned_original)
        # print("performance of pruned original")
        # pruned_original_performance = test(pruned_original, use_cuda, testloader)

        for j, sigma in enumerate(sigmas):
            # mean_index = []
            # for k in range(number_of_populations):
            pruned_original = copy.deepcopy(net)

            performance_diff = []
            stochastic_performance = []
            transfer_mask_performance = []

            # performances = [pruned_original_performance]
            # Loop over the  population
            for n in range(N):
                current_model = copy.deepcopy(net)
                if cfg.noise == "gaussian":

                    current_model.apply(partial(add_gaussian_noise_to_weights, sigma=sigma))
                elif cfg.noise == "geogaussian":

                    current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=sigma))

                prune.global_unstructured(
                    weigths_to_prune(current_model),
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_percentage)
                copy_buffers(from_net=current_model, to_net=pruned_original)

                # remove_reparametrization(current_model)

                print("Performance for model {}".format(n))

                original_transfer_mask_model_acc = test(pruned_original, use_cuda, testloader)
                stochastic_model_performance = test(current_model, use_cuda, testloader)
                performance_diff.append(original_transfer_mask_model_acc - stochastic_model_performance)
                stochastic_performance.append(stochastic_model_performance)
                transfer_mask_performance.append(original_transfer_mask_model_acc)
                # current_model.cpu()
                # performances.append(current_model_acc)
            #

            # sorted_performances = sorted(performances)
            # sorted_performances.reverse()
            # rank_original_model = sorted_performances.index(pruned_original_performance)
            # mean_index.append(rank_original_model)
            matrix_diff[j, i] = np.mean(performance_diff)
            matrix_transfer_mask_acc[j, i] = np.mean(transfer_mask_performance)
            matrix_stochastic_acc[j, i] = np.mean(stochastic_performance)
            # matrix_ranking[j, i] = np.mean(mean_index)
            print("Matrix Diff")
            print(np.matrix(matrix_diff))
            print("Matrix stochastic performance")
            print(np.matrix(matrix_stochastic_acc))
            print("Matrix mask transfer")
            print(np.matrix(matrix_transfer_mask_acc))
            # print("Matrix Ranking")
            # print(np.matrix(matrix_ranking))
        # matrix_diff[j, i] = i+j
        # matrix_ranking[j, i] = i-j
    # This is the difference between the stochastic pruned model and the mask transfer on to the original one
    df_mean_decrease = pd.DataFrame(matrix_diff, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_diff, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/bufferTransfer_V_stochasticPruned_diff_matrix_{cfg.noise}_pop"
                            f"_{cfg.population}.csv")
    #  This is the performance of the stochastic pruned model
    df_mean_decrease = pd.DataFrame(matrix_stochastic_acc, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_diff, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/stochastic_comparison_stochastic_performance_matrix_{cfg.noise}_pop"
                            f"_{cfg.population}.csv")
    # df_mean_decrease.to_csv(f"data/mean_decrease_matrix_{cfg.noise}.csv")
    #  This is the performance of the mask transfer model
    df_mean_decrease = pd.DataFrame(matrix_transfer_mask_acc, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_diff, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/stochastic_comparison_maskTransfer_performance_matrix_{cfg.noise}_pop"
                            f"_{cfg.population}.csv")
    ####################################  mean decrease heatmap #################################
    sigmas = list(map(format_sigmas, sigmas))
    ####################################  ranking of original pruned model heatmap #################################
    ax = plt.subplot()
    im = ax.imshow(matrix_diff.transpose(), aspect=0.5)
    # ax.set_xticklabels(sigmas, minor=False, rotation=-90)
    # ax.set_yticklabels(pruning_percentages, minor=False)
    plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
    plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
    plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
    plt.ylabel("Pruning Rate", fontsize=12)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Mean decrease', rotation=270, labelpad=8.5)
    # plt.gcf().set_size_inches(5.1, 5.1)
    plt.title("Mask Transfer V Stochastic pruned: {}".format(cfg.noise))
    texts = annotate_heatmap(im, valfmt="{x}")
    plt.tight_layout()
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    result = time.localtime(time.time())
    plt.savefig(f"data/figures/meanDecrease_bufferTransfer_pruning_V_{cfg.noise}Noise_{result.tm_hour}"
                f"-{result.tm_min}.pdf")
    plt.savefig(f"data/figures/meanDecrease_bufferTransfer_pruning_V_{cfg.noise}Noise_{result.tm_hour}"
                f"-{result.tm_min}.png")
    plt.close()


def heatmap2_mean_decrease_maskTransfer_exp_to_random(cfg):
    data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
                                                                               "University of Leeds\PhD\Datasets\CIFAR10"
    use_cuda = torch.cuda.is_available()
    net = None
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                             num_workers=cfg.num_workers)
    load_model(net, "trained_models/cifar10/cifar_csghmc_5.pt")
    N = cfg.population
    pruning_percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pruning_percentages.reverse()
    sigmas = None
    if cfg.noise == "gaussian":
        sigmas = np.linspace(0.001, 0.01, 10)
    if cfg.noise == "geogaussian":
        sigmas = np.linspace(0.1, 0.7, 10)
    matrix_diff = np.zeros((len(sigmas), len(pruning_percentages)))
    matrix_stochastic = np.zeros((len(sigmas), len(pruning_percentages)))
    matrix_deterministic = np.zeros((len(sigmas), len(pruning_percentages)))
    random_net_1 = ResNet18()
    for i, pruning_percentage in enumerate(pruning_percentages):

        # Here I just prune once for the deterministic pruning

        pruned_original = copy.deepcopy(net)
        weights_to_prune = weigths_to_prune(pruned_original)
        prune.global_unstructured(
            weights_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_percentage
        )
        # remove_reparametrization(pruned_original)
        deterministic_net = copy.deepcopy(random_net_1)
        copy_buffers(from_net=pruned_original, to_net=deterministic_net)
        for j, sigma in enumerate(sigmas):
            # mean_index = []
            # for k in range(number_of_populations):
            performance_diff = []
            stochastic_performance = []
            deterministic_net_performance = []
            # # performances = [pruned_original_performance]
            # # Loop over the  population
            for n in range(N):
                stochastic_random_net = copy.deepcopy(random_net_1)

                current_model = copy.deepcopy(net)
                if cfg.noise == "gaussian":

                    current_model.apply(partial(add_gaussian_noise_to_weights, sigma=sigma))
                elif cfg.noise == "geogaussian":

                    current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=sigma))

                prune.global_unstructured(
                    weigths_to_prune(current_model),
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_percentage)
                copy_buffers(from_net=current_model, to_net=stochastic_random_net)

                # remove_reparametrization(current_model)

                print("Performance for model {}".format(n))

                stochastic_random_net_acc = test(stochastic_random_net, use_cuda, testloader)
                deterministic_net_acc = test(deterministic_net, use_cuda, testloader)

                performance_diff.append(deterministic_net_acc - stochastic_random_net_acc)
                stochastic_performance.append(stochastic_random_net_acc)
                deterministic_net_performance.append(deterministic_net_acc)

            # sorted_performances = sorted(performances)
            # sorted_performances.reverse()
            # rank_original_model = sorted_performances.index(pruned_original_performance)
            # mean_index.append(rank_original_model)
            matrix_diff[j, i] = np.mean(performance_diff)
            matrix_stochastic[j, i] = np.mean(stochastic_performance)
            matrix_deterministic[j, i] = np.mean(deterministic_net_performance)

            # matrix_ranking[j, i] = np.mean(mean_index)
            print("Matrix Diff deterministic vs stochastic (det-stho)")
            print(np.matrix(matrix_diff))
            print("Matrix mean stochastic performance")
            print(np.matrix(matrix_stochastic))
            print("Matrix mean deterministic performance")
            print(np.matrix(matrix_deterministic))

            # print("Matrix Ranking")
            # print(np.matrix(matrix_ranking))
        # matrix_mean_decrease[j, i] = i+j
        # matrix_ranking[j, i] = i-j

    df_mean_decrease = pd.DataFrame(matrix_diff, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_mean_decrease, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/random_stochastic_V_deterministic_diff_matrix_{cfg.noise}_pop_{cfg.population}.csv")

    df_mean_decrease = pd.DataFrame(matrix_stochastic, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_mean_decrease, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/random_stochastic_performance_matrix_{cfg.noise}_pop_{cfg.population}.csv")

    df_mean_decrease = pd.DataFrame(matrix_deterministic, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_mean_decrease, columns=pruning_percentages)
    df_mean_decrease.index = sigmas
    # df_mean_decrease.index = sigmas
    df_mean_decrease.to_csv(f"data/random_deterministic_performance_matrix_{cfg.noise}_pop_{cfg.population}.csv")
    # df_mean_decrease.to_csv(f"data/mean_decrease_matrix_{cfg.noise}.csv")
    ####################################  mean decrease heatmap #################################
    sigmas = list(map(format_sigmas, sigmas))
    ####################################  ranking of original pruned model heatmap #################################
    ax = plt.subplot()
    im = ax.imshow(matrix_diff.transpose(), aspect=0.5)
    # ax.set_xticklabels(sigmas, minor=False, rotation=-90)
    # ax.set_yticklabels(pruning_percentages, minor=False)
    plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
    plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
    plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
    plt.ylabel("Pruning Rate", fontsize=12)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Difference in performance', rotation=270, labelpad=8.5)
    # plt.gcf().set_size_inches(5.1, 5.1)
    plt.title("Stochastic V deterministic Mask: {}".format(cfg.noise))
    texts = annotate_heatmap(im, valfmt="{x:1.2f}")
    plt.tight_layout()
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    result = time.localtime(time.time())
    plt.savefig(f"data/figures/diff_bufferTransfer_det_V_stochastic_{cfg.noise}Noise_{result.tm_hour}"
                f"-{result.tm_min}.pdf")
    plt.savefig(f"data/figures/diff_bufferTransfer_det_V_stochastic_{cfg.noise}Noise_{result.tm_hour}"
                f"-{result.tm_min}.png")
    plt.close()
    ######################################
    # ax = plt.subplot()
    # im = ax.imshow(matrix_diff.transpose(), aspect=0.5)
    # # ax.set_xticklabels(sigmas, minor=False, rotation=-90)
    # # ax.set_yticklabels(pruning_percentages, minor=False)
    # plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
    # plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
    # plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
    # plt.ylabel("Pruning Rate", fontsize=12)
    # cbar = plt.colorbar(im)
    # cbar.ax.set_ylabel('Difference in performance', rotation=270, labelpad=8.5)
    # # plt.gcf().set_size_inches(5.1, 5.1)
    # plt.title("Stochastic V deterministic Mask: {}".format(cfg.noise))
    # texts = annotate_heatmap(im, valfmt="{x:1.2f}")
    # plt.tight_layout()
    # # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # result = time.localtime(time.time())
    # plt.savefig(f"data/figures/meanDecrease_bufferTransfer_pruning_V_{cfg.noise}Noise_{result.tm_hour}"
    #             f"-{result.tm_min}.pdf")
    # plt.close()


def format_sigmas(sigma):
    string = "{:1.3f}".format(sigma)
    return string


def format_percentages(value):
    string = "{:10.1f}%".format(value)
    return string


def plot_heatmaps(cfg, plot_index=1):
    sigmas = None
    if cfg.noise == "gaussian":
        matrix_mean_decrease = np.loadtxt("data/mean_decrease_matrix_gaussian_traditional_average.txt") * 100
        matrix_ranking = np.loadtxt("data/ranking_matrix_gaussian_traditional_average.txt")
        sigmas = np.linspace(0.001, 0.01, 10)
    if cfg.noise == "geogaussian":
        matrix_mean_decrease = np.loadtxt("data/mean_decrease_mask_transfer_geogaussian.txt") * 100
        matrix_ranking = np.loadtxt("data/ranking_matrix_geogaussian.txt")
        sigmas = np.linspace(0.1, 0.7, 10)
    df_mean_decrease = pd.read_csv(f"data/ranking_matrix.csv", header=0, index_col=0)
    # sigmas = list(df_mean_decrease.index)
    sigmas = list(map(format_sigmas, sigmas))
    pruning_percentages = df_mean_decrease.columns

    # df_ranking = pd.DataFrame(matrix_ranking, columns=pruning_percentages)
    # df_mean_decrease = pd.DataFrame(matrix_ranking, columns=pruning_percentages)
    # df_ranking.index = sigmas
    # df_mean_decrease.index = sigmas
    # df_ranking.to_csv(f"data/ranking_matrix_{cfg.noise}.csv")
    # df_mean_decrease.to_csv(f"data/mean_decrease_matrix_{cfg.noise}.csv")
    if plot_index >= 0:
        ####################################  mean decrease heatmap #################################
        ax = plt.subplot()
        im = ax.imshow(matrix_mean_decrease.transpose(), aspect=0.5)
        # ax.set_xticklabels(sigmas, minor=False, rotation=-90)
        plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
        plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
        # ax.set_yticklabels(pruning_percentages, minor=False)
        # plt.yticks(ticks=pruning_percentages)
        # plt.xticks(ticks=sigmas)
        plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
        plt.ylabel("Pruning Rate", fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('Mean decrease in performance', rotation=270, labelpad=8.5)
        # plt.gcf().set_size_inches(5.1, 5.1)
        texts = annotate_heatmap(im, valfmt="{x:0.1f}", convert_to_int=False)
        plt.title("Mask transfer performance decrease", fontsize=15)
        plt.tight_layout()
        result = time.localtime(time.time())
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
        plt.savefig(f"data/figures/meanDecrease_traditional_original_V_stochastic_{cfg.noise}Noise_{result.tm_hour}"
                    f"-{result.tm_min}.png")
        plt.savefig(f"data/figures/meanDecrease_traditional_original_V_stochastic_{cfg.noise}Noise_{result.tm_hour}"
                    f"-{result.tm_min}.pdf")
        plt.close()
    ####################################  ranking of original pruned model heatmap #################################
    if plot_index > 0:
        ax = plt.subplot()
        im = ax.imshow(matrix_ranking.transpose(), aspect=0.5)
        # ax.set_xticklabels(sigmas, minor=False, rotation=-90)
        # ax.set_yticklabels(pruning_percentages, minor=False)
        plt.xticks(ticks=range(0, len(sigmas)), labels=sigmas, rotation=45)
        plt.yticks(ticks=range(0, len(pruning_percentages)), labels=pruning_percentages)
        plt.xlabel(r"$\sigma_{{}}$".format(cfg.noise), fontsize=15)
        plt.ylabel("Pruning Rate", fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('Rank', rotation=270, labelpad=8.5)
        # plt.gcf().set_size_inches(5.1, 5.1)
        plt.title("Rank of deterministic")
        texts = annotate_heatmap(im, valfmt="{x}")
        plt.tight_layout()
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        result = time.localtime(time.time())
        plt.savefig(
            f"data/figures/ranking_traditional_original_V_stochastic_{cfg.noise}Noise_{result.tm_hour}-{result.tm_min}.png")
        plt.savefig(f"data/figures/ranking_traditional_original_V_stochastic_{cfg.noise}Noise_{result.tm_hour}-"
                    f"{result.tm_min}.png")
        plt.close()


################################# Noise calibration with optuna ##################################
def calibrate(trial: optuna.trial.Trial) -> np.ndarray:
    # in theory cfg is available everywhere because it is define on the if name ==__main__ section
    net = None
    if cfg.architecture == "resnet18":
        net = ResNet18()
        load_model(net, "trained_models/cifar10/cifar_csghmc_5.pt")
    # sigma_add = trial.suggest_float("sigma_add", 0.0001, 0.1)
    sigma_add = cfg.sigma
    sigma_mul = trial.suggest_float("sigma_mul", 0.1, 1)

    # Prune original
    pruned_original = copy.deepcopy(net)
    weights_prune = weigths_to_prune(pruned_original)
    prune.global_unstructured(
        weights_prune,
        pruning_method=prune.L1Unstructured,
        amount=cfg.amount
    )
    remove_reparametrization(model=pruned_original)
    vector_original = nn.utils.parameters_to_vector(pruned_original.parameters())
    binary_vector_original = vector_original == 0
    average_loss = []
    for i in range(20):
        add_noise_model = copy.deepcopy(net)
        mul_noise_model = copy.deepcopy(net)
        add_noise_model.apply(partial(add_gaussian_noise_to_weights, sigma=sigma_add))
        mul_noise_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=sigma_mul))

        # Pruning of the  noisy models
        # Additive
        weights_prune = weigths_to_prune(add_noise_model)
        prune.global_unstructured(
            weights_prune,
            pruning_method=prune.L1Unstructured,
            amount=cfg.amount
        )
        remove_reparametrization(model=add_noise_model)
        # Geometric
        weights_prune = weigths_to_prune(mul_noise_model)
        prune.global_unstructured(
            weights_prune,
            pruning_method=prune.L1Unstructured,
            amount=cfg.amount
        )
        remove_reparametrization(model=mul_noise_model)
        #
        vector_added_noise = nn.utils.parameters_to_vector(add_noise_model.parameters())
        binary_vector_add_noise = vector_added_noise == 0
        vector_mul_noise = nn.utils.parameters_to_vector(mul_noise_model.parameters())
        binary_vector_mul_noise = vector_mul_noise == 0

        different_add_noise = torch.bitwise_xor(binary_vector_original, binary_vector_add_noise).sum()

        different_mul_noise = torch.bitwise_xor(binary_vector_original, binary_vector_mul_noise).sum()
        print(f"Number of changes by gaussian {different_add_noise}")
        print(f"Number of changes by geometric gaussian {different_mul_noise}")
        # I use the sum because I don't care where (in the vector) each noise changes specifically
        loss = ((different_add_noise - different_mul_noise) ** 2).item()
        #     average_loss = average_loss + (loss - average_loss) / (i + 1)
        average_loss.append(loss)
    print(f"Number of changes by gaussian {different_add_noise}")
    print(f"Number of changes by geometric gaussian {different_mul_noise}")

    return np.mean(average_loss)


def noise_calibration(cfg: omegaconf.DictConfig):
    # distributions = {
    #     "sigma_add": optuna.distributions.FloatDistribution(0.0001, 0.1, log=True),
    #     "sigma_mul": optuna.distributions.FloatDistribution(0.1, 1, log=True),
    # }
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner()
    )

    study = optuna.create_study(direction="minimize", pruner=pruner, study_name="noise-calibration",
                                storage="sqlite:///noise_cal_database.dep", load_if_exists=True)
    study.optimize(calibrate, n_trials=500)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("\n Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # fig1 = optuna.visualization.plot_optimization_history(study)
    # fig2 = optuna.plot_intermediate_values(study)
    # fig3 = optuna.plot_param_importances(study)
    # fig4 = optuna.contour_plot(study, params=["sigma_add", "sigma_mul"])
    #
    # fig1.savefig("data/figures/opt_history.png")
    # fig2.savefig("data/figures/intermediate_values.png")
    # fig3.savefig("data/figures/para_importances.png")
    # fig4.savefig("data/figures/contour_plot.png")


################################# Layer importance experiments ######################################
def plot_layer_experiments(model, exp_type="a"):
    layers = get_layer_dict(model)
    layers.reverse()
    layer_names, weights = zip(*layers)

    layer_names = list(layer_names)
    weights = list(weights)
    # layer_names.reverse()
    # weights.reverse()
    prunings_percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    prunings_percentages.reverse()

    if exp_type == "a":
        matrix = np.load("data/layer_exp_A.npy")
        ax = plt.subplot()
        im = ax.imshow(matrix, aspect=0.5)
        plt.yticks(ticks=range(0, len(layer_names)), labels=layer_names)
        plt.xticks(ticks=range(0, len(prunings_percentages)), labels=prunings_percentages)
        plt.xlabel("Pruning Rate", fontsize=15)
        plt.ylabel("Layer by depth (ascending order)", fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('ACCURACY', rotation=270, labelpad=8.5)
        plt.gcf().set_size_inches(5.1, 5.1)
        texts = annotate_heatmap(im, valfmt="{x}")
        plt.tight_layout()
        result = time.localtime(time.time())
        plt.savefig(f"data/figures/layer_V_prune_{result.tm_hour}-{result.tm_min}.pdf")
        plt.close()
    if exp_type == "b":
        matrix = np.loadtxt("data/layer_exp_B.txt")
        count = lambda w: w.nelement()
        number_of_elements = list(map(count, weights))
        sorted_by_n = np.argsort(number_of_elements)
        sorted_layer_names = [layers[g] for g in sorted_by_n]
        layer_names_by_n, _ = zip(*sorted_layer_names)
        ax = plt.subplot()
        im = ax.imshow(matrix, aspect=0.5)
        plt.yticks(ticks=range(0, len(layer_names_by_n)), labels=layer_names_by_n)
        plt.xticks(ticks=range(0, len(prunings_percentages)), labels=prunings_percentages)
        plt.xlabel("Pruning Rate", fontsize=15)
        plt.ylabel("Layer by # parameters \n (descending order)", fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('ACCURACY', rotation=270, labelpad=8.5)
        # cbar.ax.tick_params(labelsize=7)

        plt.gcf().set_size_inches(5.5, 5)
        texts = annotate_heatmap(im, valfmt="{x}")
        plt.tight_layout()
        result = time.localtime(time.time())
        # data / figures / cumsum.layer_V_prune_traditional_pruned_
        # {result.tm_hour} - {result.tm_min}.pdf
        # data / figures / cumsum.layer_V_prune_
        # {result.tm_hour} - {result.tm_min}.pdf
        plt.savefig(f"data / figures / cumsum.layer_V_prune_traditional_pruned_{result.tm_hour} - {result.tm_min}.pdf")
        plt.close()


def layer_importance_experiments(cfg, model, use_cuda, test_loader, type_exp="a"):
    layers = get_layer_dict(model)
    layers.reverse()
    layer_names, weights = zip(*layers)

    # layer_names = list(layer_names)
    # weights = list(weights)
    # layer_names.reverse()
    # weights.reverse()
    prunings_percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    prunings_percentages.reverse()
    matrix = np.zeros((len(layers), len(prunings_percentages)))

    if type_exp == "a":
        print("\n######################### EXPERIMENT A ##################################\n")
        for i, layer_tuple in enumerate(layers):
            for j, pruning_percentage in enumerate(prunings_percentages):
                current_model = copy.deepcopy(model)
                name, _ = layer_tuple
                weights = [get_weights_of_layer_name(current_model, layer_name=name)]
                prune.global_unstructured(
                    weights,
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_percentage
                )
                remove_reparametrization(current_model, name_module=name)
                accuracy = float(test(current_model, use_cuda=use_cuda, testloader=test_loader))
                # matrix[i, j] = int((i+j))
                matrix[i, j] = accuracy
                print(f"\n{np.matrix(matrix)}")
        np.save("data/layer_exp_A.npy", matrix)
        # matrix = np.random.randn(len(layers),len(prunings_percentages))
        ax = plt.subplot()
        im = ax.imshow(matrix, aspect=0.5)
        plt.yticks(ticks=range(0, len(layer_names)), labels=layer_names)
        plt.xticks(ticks=range(0, len(prunings_percentages)), labels=prunings_percentages)
        plt.xlabel("Pruning Rate", fontsize=15)
        plt.ylabel("Layer by depth (ascending order)", fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('ACCURACY', rotation=270, labelpad=8.5)
        plt.gcf().set_size_inches(5.1, 5.1)
        texts = annotate_heatmap(im, valfmt="{x}")
        plt.tight_layout()
        result = time.localtime(time.time())
        plt.savefig(f"data/figures/traditional_trained_layer_V_prune_{result.tm_hour}-{result.tm_min}.pdf")
        plt.close()
        percent_index = matrix.argmax(axis=1)
        best_percentage_for_layers = {}
        for i, name in enumerate(layer_names):
            best_percentage_for_layers[name] = prunings_percentages[percent_index[i]]
        print(f"Best percentage by layer {best_percentage_for_layers}")
        with open(f"data/best_per_layer_{cfg.architecture}_traditional_solution.pkl", "wb") as f:
            pickle.dump(best_percentage_for_layers, f)

    if type_exp == "b":
        # Organise the layers to accumulate with the number of weights
        # with open(f"data/best_per_layer_{cfg.architecture}.pkl", "rb") as f:
        #     prune_rate_per_layer = pickle.load(f)
        #
        print("\n######################### EXPERIMENT B ##################################\n")
        count = lambda w: w.nelement()
        number_of_elements = list(map(count, weights))
        sorted_by_n = np.argsort(number_of_elements)
        sorted_layer_names = [layers[g] for g in sorted_by_n]
        # sorted_layer_names
        # Then im going to prune all the layers up to layer i with the pruning rate obtained from the type a experiment.
        for i, layer_tuple in enumerate(sorted_layer_names):
            for j, pruning_percentage in enumerate(prunings_percentages):
                current_model = copy.deepcopy(model)

                sub_layers = sorted_layer_names[i:]
                weights = [get_weights_of_layer_name(current_model, layer_name=n) for n, w in sub_layers]
                # name, _ = layer_tuple
                # weights = get_weights_of_layer_name(current_model, layer_name=name)
                prune.global_unstructured(
                    weights,
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_percentage
                )
                for n, w in sub_layers:
                    remove_reparametrization(current_model, name_module=n)
                accuracy = float(test(current_model, use_cuda=use_cuda, testloader=test_loader))
                # matrix[i, j] = int((i+j))
                matrix[i, j] = accuracy
                print(f"\n{np.matrix(matrix)}")
        np.save("data/layer_exp_B.npy", matrix)
        count = lambda w: w.nelement()
        number_of_elements = list(map(count, weights))
        sorted_by_n = np.argsort(number_of_elements)
        sorted_layer_names = [layers[g] for g in sorted_by_n]
        layer_names_by_n, _ = zip(*sorted_layer_names)
        ax = plt.subplot()
        im = ax.imshow(matrix, aspect=0.5)
        plt.yticks(ticks=range(0, len(layer_names_by_n)), labels=layer_names_by_n)
        plt.xticks(ticks=range(0, len(prunings_percentages)), labels=prunings_percentages)
        plt.xlabel("Pruning Rate", fontsize=15)
        plt.ylabel("Layer by # parameters \n (descending order)", fontsize=12)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('ACCURACY', rotation=270, labelpad=8.5)
        # cbar.ax.tick_params(labelsize=7)

        plt.gcf().set_size_inches(5.5, 5)
        texts = annotate_heatmap(im, valfmt="{x}")
        plt.tight_layout()
        result = time.localtime(time.time())
        plt.savefig(f"data/figures/cumsum.layer_V_prune_traditional_pruned_{result.tm_hour}-{result.tm_min}.pdf")
        plt.close()
    #     # Now I'm going to do something similar to the above but just prune with the optimal rate


def run_layer_experiment(cfg):
    use_cuda = torch.cuda.is_available()
    if cfg.architecture == "resnet18" and "csghmc" in cfg.solution:
        net = ResNet18()
    else:
        from alternate_models import ResNet18
        net = ResNet18()
    if cfg.use_wandb:
        # os.environ["WANDB_START_METHOD"] = "thread"
        # now = date.datetime.now().strftime("%M:%S")
        # wandb.init(
        #     entity="luis_alfredo",
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     project="sparse_ensemble",
        #     name="layer_importance_{}".format(now),
        #     reinit=True,
        #     save_code=True,
        # )
        pass
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    load_model(net, cfg.solution)

    # plot_layer_experiments(net, exp_type="a")
    # layer_importance_experiments(cfg, net, use_cuda, testloader, type_exp="a")
    # plot_layer_experiments(net, exp_type="a")
    # layer_importance_experiments(cfg, net, use_cuda, testloader, type_exp="b")
    plot_layer_experiments(net, exp_type="b")


def get_best_pruning_strategy(cfg: omegaconf.DictConfig):
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner()
    )
    ####################### This section does compute all layer combinations of the small layers that ResNet18 has.j
    # net = None
    # if cfg.architecture == "resnet18" and "csgmcmc" in cfg.solution:
    #     net = ResNet18()
    # else:
    #     from alternate_models.resnet import ResNet18
    #     net = ResNet18()
    # load_model(net, cfg.solution)
    # layers = get_layer_dict(net)
    # layers.reverse()
    # layer_names, weights = zip(*layers)
    # # fifth element
    # up_until = 4
    # count = lambda w: w.nelement()
    # number_of_elements = list(map(count, weights))
    # sorted_by_n = np.argsort(number_of_elements)
    # sorted_layer_names = [layers[g] for g in sorted_by_n]
    # high_pruned = sorted_layer_names[:up_until + 1]
    # important_layer = "layer3.1.conv1"
    # not_high_pruned = dict(set(sorted_layer_names).difference(set(high_pruned)))
    # del not_high_pruned[important_layer]
    # all_groups = all_subsets(list(not_high_pruned.keys()))
    # with open(f"data/groups_small_layers_{cfg.architecture}.plk", "wb") as f:
    #     pickle.dump(tuple(all_groups), f)

    with open(f"data/groups_small_layers_{cfg.architecture}.plk", "rb") as f:
        all_groups = pickle.load(f)
    ################################################################

    study = optuna.create_study(directions=["maximize", "maximize"], pruner=pruner, study_name="Smart pruned")
    study.optimize(partial(optim_function_intelligent_pruning, cfg=cfg), n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("\n Best trial:")
    trials = study.best_trials
    with open(f"data/best_trials_smart_pruning_{cfg.architecture}.plk", "wb") as f:
        pickle.dump(trials, f)

    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[0])
    print(f"Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_highest_accuracy.number}")
    print(f"\tparams: {trial_with_highest_accuracy.params}")
    print(f"\tvalues: {trial_with_highest_accuracy.values}")
    trial_with_highest_sparsity = max(study.best_trials, key=lambda t: t.values[1])
    print(f"Trial with highest sparsity: ")
    print(f"\tnumber: {trial_with_highest_sparsity.number}")
    print(f"\tparams: {trial_with_highest_sparsity.params}")
    print(f"\tvalues: {trial_with_highest_sparsity.values}")

    optuna.visualization.plot_pareto_front(study, target_names=["Accuracy", "Compression ratio"])


def get_intelligent_pruned_network(cfg):
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18" and "csgmcmc" in cfg.solution:
        net = ResNet18()
    else:
        from alternate_models.resnet import ResNet
        net = ResNet18()
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    load_model(net, cfg.solution)
    layers = get_layer_dict(net)
    layers.reverse()
    layer_names, weights = zip(*layers)
    pr = 0.9
    # fifth element
    up_until = 4
    count = lambda w: w.nelement()
    number_of_elements = list(map(count, weights))
    sorted_by_n = np.argsort(number_of_elements)
    sorted_layer_names = [layers[g] for g in sorted_by_n]
    # I reverse the layer in order to get the layers with the majority of the parameters first.
    sorted_layer_names.reverse()

    base_model = copy.deepcopy(net)
    high_pruned = sorted_layer_names[:up_until + 1]
    weights = [get_weights_of_layer_name(base_model, layer_name=n) for n, w in high_pruned]
    prune.global_unstructured(
        weights,
        pruning_method=prune.L1Unstructured,
        amount=pr
    )

    for n, w in high_pruned:
        remove_reparametrization(base_model, name_module=n)
    print("pruning the following layers with {} pruning rate".format(pr))
    for name in high_pruned:
        print(f"\t{name}")
    base_model_accuracy = test(base_model, use_cuda, testloader)
    print(f"Accuracy of the base model: {base_model_accuracy}")
    print(f"Base model has compression ratio of {count_zero_parameters(base_model) / count_parameters(base_model)}")
    print("Now i'm going to prune all the other layers combined")
    important_layer = "layer3.1.conv1"
    not_high_pruned = dict(set(sorted_layer_names).difference(set(high_pruned)))
    del not_high_pruned[important_layer]
    pruning_rates = [0.4]
    for pr in pruning_rates:
        full_model = copy.deepcopy(base_model)
        weights = [get_weights_of_layer_name(full_model, layer_name=n) for n, w in not_high_pruned.items()]
        prune.global_unstructured(
            weights,
            pruning_method=prune.L1Unstructured,
            amount=pr
        )
        print(f"Pruning rate {pr} for layers {not_high_pruned}")
        test_acc = test(full_model, use_cuda, testloader)
        for n, w in not_high_pruned.items():
            remove_reparametrization(full_model, name_module=n)
        if pr == 0.4:
            torch.save(full_model.state_dict(), "trained_models/cifar10/smart_pruned.pth")
        print("Test performance for the previous pruned model")
        print(f"{test_acc}")
        print(f"Model has compression ratio of {count_zero_parameters(full_model) / count_parameters(full_model)}")


def optim_function_intelligent_pruning(trial: optuna.trial.Trial, cfg) -> tuple:
    # Pruning rate large layers
    pruning_large_layers = trial.suggest_float("Pr_big_layers", 0.8, 0.99)
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18" and "csgmcmc" in cfg.solution:
        net = ResNet18()
    else:
        from alternate_models.resnet import ResNet18
        net = ResNet18()
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    load_model(net, cfg.solution)
    n = cfg.population
    layers = get_layer_dict(net)
    layers.reverse()
    layer_names, weights = zip(*layers)
    # fifth element
    up_until = 4
    count = lambda w: w.nelement()
    number_of_elements = list(map(count, weights))
    sorted_by_n = np.argsort(number_of_elements)
    sorted_layer_names = [layers[g] for g in sorted_by_n]
    sorted_layer_names.reverse()
    base_model = copy.deepcopy(net)
    high_pruned = sorted_layer_names[:up_until + 1]
    weights = [get_weights_of_layer_name(base_model, layer_name=n) for n, w in high_pruned]
    prune.global_unstructured(
        weights,
        pruning_method=prune.L1Unstructured,
        amount=pruning_large_layers
    )
    for n, w in high_pruned:
        remove_reparametrization(base_model, name_module=n)
    # # Now I need to select how many layers of the other layers I'm going to group to prune
    # important_layer = "layer3.1.conv1"
    # not_high_pruned =dict( set(sorted_layer_names).difference(set(high_pruned)))
    # del not_high_pruned[important_layer]
    # all_groups = all_subsets(list(not_high_pruned.values()))
    with open(f"data/groups_small_layers_{cfg.architecture}.plk", "rb") as f:
        all_groups = pickle.load(f)
    # Layers Im going to acctually prune
    index = trial.suggest_int("all_groups index", 0, len(all_groups))
    layers_to_prune = all_groups[index]
    pr_for_this = trial.suggest_float("Pr_small_layers", 0.2, 0.7)
    weights = [get_weights_of_layer_name(base_model, layer_name=n) for n in layers_to_prune]
    prune.global_unstructured(
        weights,
        pruning_method=prune.L1Unstructured,
        amount=pr_for_this
    )
    for n in layers_to_prune:
        remove_reparametrization(base_model, name_module=n)
    test_accuracy = test(base_model, use_cuda, testloader)

    return test_accuracy, count_zero_parameters(base_model) / count_parameters(base_model)


######################################################################################################
def get_cifar_datasets(cfg):
    data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
                                                                               "University of Leeds\PhD\Datasets\CIFAR10"
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    cifar10_train, cifar10_val = random_split(trainset, [45000, 5000])

    trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=cfg.batch_size, shuffle=True,
                                              num_workers=cfg.num_workers)
    val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=cfg.batch_size, shuffle=True,
                                             num_workers=cfg.num_workers)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                             num_workers=cfg.num_workers)
    return trainloader, val_loader, testloader


def main(cfg: omegaconf.DictConfig):
    print("torch version: {}".format(torch.__version__))
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18" and "csgmcmc" in cfg.solution:
        net = ResNet18()
    else:
        from alternate_models.resnet import ResNet18
        net = ResNet18()
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    state_dict = torch.load(cfg.solution)
    load_model(net, cfg.solution)
    N = cfg.population
    pop = []
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []
    original_performance = test(net, use_cuda, testloader)
    pruned_original = copy.deepcopy(net)
    weights_to_prune = weigths_to_prune(pruned_original)
    prune.global_unstructured(
        weights_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=cfg.amount
    )
    remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(pruned_original, use_cuda, testloader)
    pop.append(pruned_original)
    pruned_performance.append(pruned_original_performance)
    stochastic_dense_performances.append(original_performance)
    # pp = pprint.PrettyPrinter(ident=4)
    for n in range(N):
        current_model = copy.deepcopy(net)
        if cfg.noise == "gaussian":
            current_model.apply(partial(add_gaussian_noise_to_weights, sigma=cfg.sigma))
        elif cfg.noise == "geogaussian":
            current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=cfg.sigma))
        print("Stochastic dense performance")
        StoDense_performance = test(current_model, use_cuda, testloader)

        # Dense stochastic performance
        stochastic_dense_performances.append(StoDense_performance)
        prune.global_unstructured(
            weigths_to_prune(current_model),
            pruning_method=prune.L1Unstructured,
            amount=cfg.amount)
        remove_reparametrization(current_model)
        print("Performance for model {}".format(n))
        stochastic_pruned_performance = test(current_model, use_cuda, testloader)
        pruned_performance.append(stochastic_pruned_performance)
        stochastic_deltas.append(StoDense_performance - stochastic_pruned_performance)
        current_model.cpu()
        pop.append(current_model)
    # proba_function = get_proba_function(cfg.amount, count_parameters(net))
    ranked_index = np.flip(np.argsort(pruned_performance))
    index_of_pruned_original = list(ranked_index).index(0)
    pruned_performance = np.array(pruned_performance)
    stochastic_dense_performances = np.array(stochastic_dense_performances)
    np.save("data/population_data/performances_{}.npy".format(cfg.noise), pruned_performance)
    for i, model in enumerate(pop):
        with open("data/population_data/{}/model_{}.pkl".format(cfg.noise, i), "wb") as f:
            pickle.dump(model, f)
    del pop
    cutoff = original_performance - 2

    fig = plt.figure()
    plt.axhline(y=original_performance, color="k", linestyle="-", label="Original dense pruned_performance")
    plt.axhline(y=cutoff, color="r", linestyle="--", label="cutoff value")
    plt.xlabel("Ranking index", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    if cfg.noise == "geogaussian":
        plt.title("CIFAR10 Geometric Gaussian Noise", fontsize=20)

    if cfg.noise == "gaussian":
        plt.title("CIFAR10 Additive Gaussian Noise", fontsize=20)

    stochastic_models_points_dense = []
    stochastic_models_points_pruned = []
    p1 = None
    for i, element in enumerate(pruned_performance[ranked_index]):
        if i == index_of_pruned_original:
            p1 = plt.scatter(i, element, c="g", marker="o")
        else:
            pruned_point = plt.scatter(i, element, c="b", marker="x")
            stochastic_models_points_pruned.append(pruned_point)
    for i, element in enumerate(stochastic_dense_performances[ranked_index]):
        if i == index_of_pruned_original:
            continue
            # plt.scatter(i, element, c="y", marker="o", label="original model performance")
        else:
            dense_point = plt.scatter(i, element, c="c", marker="1")
            stochastic_models_points_dense.append(dense_point)

    plt.legend([tuple(stochastic_models_points_pruned), tuple(stochastic_models_points_dense), p1],
               ['Pruned stochastic', 'Dense stochastic', "original model pruned"],
               scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)})
    plt.tight_layout()
    result = time.localtime(time.time())
    plt.savefig(
        f"data/figures/comparison_{cfg.noise}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop_{cfg.population}_{result.tm_hour}"
        f"-{result.tm_min}.pdf")
    plt.savefig(f"data/figures/comparison_{cfg.noise}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop_{cfg.population}"
                f"_{result.tm_hour}"
                f"-{result.tm_min}.png")


def plot(cfg):
    performance = np.load("data/population_data/performances_{}.npy".format(cfg.noise))
    ranked_index = np.flip(np.argsort(performance))
    cutoff = 92
    plt.figure()
    plt.axhline(y=94.87, color="k", linestyle="-", label="Dense performance")
    plt.axhline(y=cutoff, color="r", linestyle="--", label="cutoff value")
    plt.xlabel("Ranking index", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    maximum = np.max(performance)
    for i, element in enumerate(performance[ranked_index]):
        if element == maximum:
            plt.scatter(i, element, c="g", marker="o", label="original model pruned")
        else:
            plt.scatter(i, element, c="b", marker="x")
    if cfg.noise == "geogaussian":
        plt.title("CIFAR10 Geometric Gaussian Noise", fontsize=20)

    if cfg.noise == "gaussian":
        plt.title("CIFAR10 Additive Gaussian Noise", fontsize=20)
    plt.legend()
    plt.show()


def manual_train(model: nn.Module, train_loader, val_loader, save_name, epochs, learning_rate, is_cyclic=False,
                 cosine_schedule=False,
                 lr_peak_epoch=5, weight_decay=1e-4, momentum=0.9, grad_clip=0.1, optim="sgd", solution_accuracy=0):
    model.cuda()
    if optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                                    nesterov=True)
    elif optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
    iters_per_epoch = len(train_loader)
    lr_scheduler = np.interp(np.arange((epochs + 1) * iters_per_epoch),
                             [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                             [0, 1, 0])
    if is_cyclic:
        # lr_scheduler = LambdaLR(optimizer, lr_schedule.__getitem__)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=epochs,
                                                           steps_per_epoch=len(train_loader))

    elif cosine_schedule:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:

        lr_scheduler = ExponentialLR(optimizer, gamma=0.90)

    # trainer = create_supervised_trainer(model, optimizer, criterion, device="cuda")
    val_metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }
    accuracy = Accuracy()

    if solution_accuracy:
        best_accuracy = solution_accuracy
    else:
        best_accuracy = 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # first forward-backward step
            predictions = model(data)
            # enable_bn(model)
            loss = criterion(predictions, target)
            loss.backward()
            accuracy.update((predictions, target))
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            lr_scheduler.step()
            # smooth_CE(model(data), target).backward()
            # optimizer.second_step(zero_grad=True)
            # if mask:
            #     mask.apply_mask()
            #     one_forward_backward_pass = mask.inference_FLOPs * 3
            #     train_flops += one_forward_backward_pass * 2
            # # L2 Regularization
            #
            # # Exp avg collection
            # _loss_collector.add_value(loss.item())
            #
            # pbar.update(1)
            # global_step += 1

            if batch_idx % 10 == 0:
                acc = accuracy.compute()
                print(f"Training Results - Epoch: {epoch}  Avg accuracy: {acc:.2f} Avg loss:"
                      f" {loss.item():.2f}")

        print("Validation results")
        performance = test(model, use_cuda=True, testloader=val_loader)
        msg_performance = f"{performance:.2f}".replace(".", ",")
        if performance > best_accuracy:
            torch.save(model.state_dict(), f"trained_models/{save_name}_val_accuracy={msg_performance}.pt")


def run_traditional_training(cfg):
    data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
                                                                               "University of Leeds\PhD\Datasets\CIFAR10"
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18":
        if cfg.solution:
            net = tv.models.resnet18()
            net.fc = nn.Linear(512, 10)
            load_model(net, cfg.solution)
        else:
            from torchvision.models import resnet18, ResNet18_Weights
            net = tv.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            net.fc = nn.Linear(512, 10)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    cifar10_train, cifar10_val = random_split(trainset, [45000, 5000])

    trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=cfg.batch_size, shuffle=True,
                                              num_workers=cfg.num_workers)
    val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=cfg.batch_size, shuffle=True,
                                             num_workers=cfg.num_workers)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                             num_workers=cfg.num_workers)

    pre_train_perf = test(net, use_cuda, testloader)
    manual_train(net, trainloader, val_loader, save_name=f"traditional_trained_cylic_{cfg.cyclic_lr}_cosine_"
                                                         f"{cfg.cosineschedule}_optim_{cfg.optim}",
                 epochs=cfg.epochs,
                 learning_rate=cfg.lr,
                 is_cyclic=cfg.cyclic_lr, cosine_schedule=cfg.cosine_schedule, lr_peak_epoch=cfg.lr_peak_epoch,
                 optim=cfg.optim, solution_accuracy=pre_train_perf if cfg.solution else 0)
    print("TEST SET RESULTS")
    test(net, use_cuda, testloader)


def train(model: nn.Module, train_loader, val_loader, save_name, epochs, learning_rate, is_cyclic=False,
          cosine_schedule=False, lr_peak_epoch=5, weight_decay=1e-4, momentum=0.9):
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                                nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
    iters_per_epoch = len(train_loader)
    lr_schedule = np.interp(np.arange((epochs + 1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    if is_cyclic:
        # lr_scheduler = LambdaLR(optimizer, lr_schedule.__getitem__)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=epochs,
                                                           steps_per_epoch=len(train_loader))
    else:
        lr_scheduler = ExponentialLR(optimizer, gamma=0.90)

    trainer = create_supervised_trainer(model, optimizer, criterion, device="cuda")
    val_metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device="cuda")

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    print("\nFine tuning has began\n")

    # Setup engine &  logger
    def setup_logger(logger):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, TerminateOnNan

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # Store the best model
    def default_score_fn(engine):
        score = engine.state.metrics['accuracy']
        return score

    # Force filename to model.pt to ease the rerun of the notebook
    disk_saver = DiskSaver(dirname="trained_models", require_empty=False)
    best_model_handler = Checkpoint(to_save={f'{save_name}': model},
                                    save_handler=disk_saver,
                                    filename_pattern="{name}.{ext}",
                                    n_saved=1)
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

    # Add early stopping
    es_patience = 10
    es_handler = EarlyStopping(patience=es_patience, score_function=default_score_fn, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, es_handler)
    setup_logger(es_handler.logger)

    # Clear cuda cache between training/testing
    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())
    trainer.run(train_loader, max_epochs=epochs)


def get_stochastic_pruning_results_on(cfg):
    pass


############################### Experiments 25 of October # ############################################################

def transfer_mask_rank_experiments(cfg: omegaconf.DictConfig):
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18" and "csgmcmc" in cfg.solution:
        net = ResNet18()
    else:
        from alternate_models.resnet import ResNet18
        net = ResNet18()

    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    load_model(net, cfg.solution)
    N = cfg.population
    pop = []
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []
    deterministic_with_stochastic_mask_performance = []
    stochastic_with_deterministic_mask_performance = []

    original_performance = test(net, use_cuda, testloader)
    pruned_original = copy.deepcopy(net)
    weights_to_prune = weigths_to_prune(pruned_original)
    prune.global_unstructured(
        weights_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=cfg.amount
    )
    # remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(pruned_original, use_cuda, testloader)
    pop.append(pruned_original)
    pruned_performance.append(pruned_original_performance)
    labels = ["pruned original"]
    stochastic_dense_performances.append(original_performance)
    for n in range(N):
        current_model = copy.deepcopy(net)
        sto_mask_transfer_model = copy.deepcopy(net)
        if cfg.noise == "gaussian":
            current_model.apply(partial(add_gaussian_noise_to_weights, sigma=cfg.sigma))
        elif cfg.noise == "geogaussian":
            current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=cfg.sigma))
        det_mask_transfer_model = copy.deepcopy(current_model)
        copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, testloader)
        stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        print("Stochastic dense performance")
        StoDense_performance = test(current_model, use_cuda, testloader)
        # Dense stochastic performance
        stochastic_dense_performances.append(StoDense_performance)
        prune.global_unstructured(
            weigths_to_prune(current_model),
            pruning_method=prune.L1Unstructured,
            amount=cfg.amount)
        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model)
        print("Performance for model {}".format(n))

        stochastic_pruned_performance = test(current_model, use_cuda, testloader)
        deterministic_with_stochastic_mask_performance.append(test(sto_mask_transfer_model, use_cuda, testloader))
        pruned_performance.append(stochastic_pruned_performance)
        stochastic_deltas.append(StoDense_performance - stochastic_pruned_performance)
        current_model.cpu()
        pop.append(current_model)
    # len(pruned performance)-1 because the first one is the pruned original
    labels.extend(["stochastic pruned"] * (len(pruned_performance) - 1))
    labels.extend(["sto mask transfer"] * len(deterministic_with_stochastic_mask_performance))
    pruned_performance.extend(deterministic_with_stochastic_mask_performance)
    stochastic_dense_performances.extend([1] * len(deterministic_with_stochastic_mask_performance))
    # Deterministic mask transfer to stochastic model
    labels.extend(["det mask transfer"] * len(stochastic_with_deterministic_mask_performance))
    pruned_performance.extend(stochastic_with_deterministic_mask_performance)
    stochastic_dense_performances.extend([1] * len(stochastic_with_deterministic_mask_performance))

    # This gives a list of the INDEXES that would sort "pruned_performance". I know that the index 0 of
    # pruned_performance is the pruned original. Then I ask ranked index where is the element 0 which references the
    # index 0 of pruned_performance.
    assert len(labels) == len(pruned_performance), f"The labels and the performances are not the same length: " \
                                                   f"{len(labels)}!={len(pruned_performance)}"
    ranked_index = np.flip(np.argsort(pruned_performance))
    index_of_pruned_original = list(ranked_index).index(0)

    pruned_performance = np.array(pruned_performance)
    stochastic_dense_performances = np.array(stochastic_dense_performances)
    result = time.localtime(time.time())
    np.save("data/population_data/performances_{}_transfer_t_{}-{}.npy".format(cfg.noise,result.tm_hour,result.tm_min),
            pruned_performance)
    # for i, model in enumerate(pop):
    #     with open("data/population_data/{}/model_{}.pkl".format(cfg.noise, i), "wb") as f:
    #         pickle.dump(model, f)
    del pop
    cutoff = original_performance - 2

    fig = plt.figure()
    plt.axhline(y=original_performance, color="k", linestyle="-", label="Original dense pruned_performance")
    plt.axhline(y=cutoff, color="r", linestyle="--", label="cutoff value")
    plt.xlabel("Ranking index", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    sigma = format_sigmas(cfg.sigma)
    pr = format_percentages(cfg.amount)
    if cfg.noise == "geogaussian":
        plt.title(f"CIFAR10 Geometric Gaussian Noise $\sigma={sigma},pr={pr}$", fontsize=10)

    if cfg.noise == "gaussian":
        plt.title(f"CIFAR10 Additive Gaussian Noise $\sigma={sigma},pr={pr}$", fontsize=10)

    stochastic_models_points_dense = []
    stochastic_models_points_pruned = []
    transfer_mask_models_points = []
    stochastic_with_deterministic_mask_models_points = []

    p1 = None

    for i, element in enumerate(pruned_performance[ranked_index]):
        if i == index_of_pruned_original:
            assert element == pruned_original_performance, "The supposed pruned original is not the original: element " \
                                                           f"in list {element} VS pruned performance:" \
                                                           f" {pruned_original_performance}"
            p1 = plt.scatter(i, element, c="g", marker="o")
        else:
            if labels[ranked_index[i]] == "sto mask transfer":
                sto_transfer_point = plt.scatter(i, element, c="tab:orange", marker="P")
                transfer_mask_models_points.append(sto_transfer_point)
            elif labels[ranked_index[i]] == "det mask transfer":
                det_transfer_point = plt.scatter(i, element, c="tab:olive", marker="X")
                stochastic_with_deterministic_mask_models_points.append(det_transfer_point)
            else:
                pruned_point = plt.scatter(i, element, c="b", marker="x")
                stochastic_models_points_pruned.append(pruned_point)
    for i, element in enumerate(stochastic_dense_performances[ranked_index]):
        if i == index_of_pruned_original or element == 1:
            continue
            # plt.scatter(i, element, c="y", marker="o", label="original model performance")
        else:
            dense_point = plt.scatter(i, element, c="c", marker="1")
            stochastic_models_points_dense.append(dense_point)

    plt.legend([tuple(stochastic_models_points_pruned), tuple(stochastic_models_points_dense), p1,
                tuple(transfer_mask_models_points), tuple(stochastic_with_deterministic_mask_models_points)],
               ['Pruned stochastic', 'Dense stochastic', "Original model pruned", "stochastic mask " + r"$\rightarrow$"
                + " original weights", "deterministic mask" + r"$\rightarrow$" + "stochastic weights"],
               scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)})
    plt.tight_layout()
    plt.savefig(
        f"data/figures/transfers_comparison_{cfg.noise}_sigma_"
        f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
        f"_{cfg.population}"
        f"_t_{result.tm_hour}"
        f"-{result.tm_min}.pdf")
    plt.savefig(f"data/figures/transfers_comparison_{cfg.noise}_sigma_{cfg.sigma}_pr_{cfg.amount}_batchSize"
                f"_{cfg.batch_size}_pop"
                f"_{cfg.population}"
                f"_t_{result.tm_hour}"
                f"-{result.tm_min}.png")


def transfer_mask_rank_experiments_no_plot(cfg: omegaconf.DictConfig):
    use_cuda = torch.cuda.is_available()
    net = None
    if cfg.architecture == "resnet18" and "csgmcmc" in cfg.solution:
        net = ResNet18()
    else:
        from alternate_models.resnet import ResNet18
        net = ResNet18()

    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    load_model(net, cfg.solution)
    N = cfg.population
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []
    sto_mask_to_ori_weights_deltas = []
    ori_mask_to_sto_weights_deltas = []

    original_with_stochastic_mask_performance = []
    stochastic_with_deterministic_mask_performance = []

    original_performance = test(net, use_cuda, testloader)
    pruned_original = copy.deepcopy(net)
    weights_to_prune = weigths_to_prune(pruned_original)
    prune.global_unstructured(
        weights_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=cfg.amount
    )
    # remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(pruned_original, use_cuda, testloader)
    pruned_performance.append(pruned_original_performance)
    labels = ["pruned original"]
    stochastic_dense_performances.append(original_performance)
    for n in range(N):
        current_model = copy.deepcopy(net)
        sto_mask_transfer_model = copy.deepcopy(net)
        if cfg.noise == "gaussian":
            current_model.apply(partial(add_gaussian_noise_to_weights, sigma=cfg.sigma))
        elif cfg.noise == "geogaussian":
            current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=cfg.sigma))
        det_mask_transfer_model = copy.deepcopy(current_model)
        copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, testloader)
        stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        print("Stochastic dense performance")
        StoDense_performance = test(current_model, use_cuda, testloader)
        # Dense stochastic performance
        stochastic_dense_performances.append(StoDense_performance)
        prune.global_unstructured(
            weigths_to_prune(current_model),
            pruning_method=prune.L1Unstructured,
            amount=cfg.amount)
        # Here is where I transfer the mask from the prunec stochastic model to the
        # original weights and put it in the ranking
        copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model)
        print("Performance for model {}".format(n))

        stochastic_pruned_performance = test(current_model, use_cuda, testloader)
        original_with_sto_mask = test(sto_mask_transfer_model, use_cuda, testloader)
        original_with_stochastic_mask_performance.append(original_with_sto_mask)
        pruned_performance.append(stochastic_pruned_performance)
        stochastic_deltas.append(original_performance-stochastic_pruned_performance)
        sto_mask_to_ori_weights_deltas.append(original_performance - original_with_sto_mask)
        ori_mask_to_sto_weights_deltas.append(original_performance-det_mask_transfer_model_performance)
    # len(pruned performance)-1 because the first one is the pruned original
    labels.extend(["stochastic pruned"] * (len(pruned_performance) - 1))
    labels.extend(["sto mask transfer"] * len(original_with_stochastic_mask_performance))
    pruned_performance.extend(original_with_stochastic_mask_performance)
    stochastic_dense_performances.extend([1] * len(original_with_stochastic_mask_performance))
    # Deterministic mask transfer to stochastic model
    labels.extend(["det mask transfer"] * len(stochastic_with_deterministic_mask_performance))
    pruned_performance.extend(stochastic_with_deterministic_mask_performance)
    stochastic_dense_performances.extend([1] * len(stochastic_with_deterministic_mask_performance))

    # This gives a list of the INDEXES that would sort "pruned_performance". I know that the index 0 of
    # pruned_performance is the pruned original. Then I ask ranked index where is the element 0 which references the
    # index 0 of pruned_performance.
    assert len(labels) == len(pruned_performance), f"The labels and the performances are not the same length: " \
                                                   f"{len(labels)}!={len(pruned_performance)}"
    ranked_index = np.flip(np.argsort(pruned_performance))
    index_of_pruned_original = list(ranked_index).index(0)

    pruned_performance = np.array(pruned_performance)
    stochastic_dense_performances = np.array(stochastic_dense_performances)
    result = time.localtime(time.time())
    np.save(
        "data/population_data/performances_{}_transfer_t_{}-{}.npy".format(cfg.noise, result.tm_hour, result.tm_min),
        pruned_performance)
    with open("data/population_data/labels_{}_transfer_t_{}-{}.npy".format(cfg.noise, result.tm_hour, result.tm_min), "wb") as f:
        pickle.dump(labels, f)

    np.save(
        "data/population_data/deltas_{}_sto_mask_to_ori_weights_deltas_N_{}_t_{}-{}.npy".format(cfg.noise,cfg.population, result.tm_hour,
                                                                          result.tm_min),
        np.array(sto_mask_to_ori_weights_deltas)
        )

    np.save(
        "data/population_data/deltas_{}_det_mask_to_sto_weights_deltas_N_{}_t_{}-{}.npy".format(cfg.nois,
                                                                                               cfg.population, result.tm_hour,
                                                                                                result.tm_min),
        np.array(ori_mask_to_sto_weights_deltas)
    )


def sigma_sweeps_transfer_mask_rank_experiments():
    cfg = omegaconf.DictConfig({
        "population": 10,
        "architecture": "resnet18",
        "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
        "model": "5",
        "noise": "gaussian",
        # "sigma": 0.0021419609859022197,
        "sigma": 0.001,
        "amount": 0.8,
        "batch_size": 512,
        "num_workers": 1,
        "use_wandb": True
    })
    sigmas = np.linspace(start=0.001, stop=0.005, num=10)
    pruning_rates = [0.5,0.6,0.8,0.83,0.85,0.87,0.9,0.95]
    for sig in sigmas:
        for pr in pruning_rates:

            cfg.sigma = float(sig)
            cfg.pr = pr
            transfer_mask_rank_experiments(cfg)


def population_sweeps_transfer_mask_rank_experiments():
    cfg = omegaconf.DictConfig({
        "population": 10,
        "architecture": "resnet18",
        "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
        "model": "5",
        "noise": "gaussian",
        "sigma": 0.0021419609859022197,
        # "sigma": 0.001,
        "amount": 0.8,
        "batch_size": 512,
        "num_workers": 1,
        "use_wandb": True
    })
    Ns = [1000]
    for pop in Ns:
        cfg.population = pop
        transfer_mask_rank_experiments_no_plot(cfg)
if __name__ == '__main__':
    # cfg_training = omegaconf.DictConfig({
    #     "architecture": "resnet18",
    #     "batch_size": 512,
    #     "lr": 0.001,
    #     "momentum": 0.9,
    #     "weight_decay": 1e-4,
    #     "cyclic_lr": True,
    #     "lr_peak_epoch": 5,
    #     "optim": "adam",
    #     "solution": "trained_models/traditional_trained_val_accuracy=91,86.pt",
    #     "num_workers": 1,
    #     "cosine_schedule": False,
    #     "epochs": 24
    # })
    # run_traditional_training(cfg_training)

    # print("GEOMETRIC GAUSSIAN NOISE")

    # print("GAUSSIAN NOISE")
    # cfg = omegaconf.DictConfig({
    #     "population": 10,
    #     "architecture": "resnet18",
    #     "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
    #     "model": "5",
    #     "noise": "gaussian",
    #     # "sigma": 0.0021419609859022197,
    #     "sigma": 0.001,
    #     "amount": 0.90,
    #     "batch_size": 512,
    #     "num_workers": 1,
    #     "use_wandb": True
    # })
    population_sweeps_transfer_mask_rank_experiments()
    # sigma_sweeps_transfer_mask_rank_experiments()
    # run_layer_experiment(cfg)
    # get_intelligent_pruned_network(cfg)
    # get_best_pruning_strategy(cfg)
    # heatmap1_exp(cfg)
    # pruning_rates = [0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    # for pr in pruning_rates:
    #     cfg.amount = pr
    #     main(cfg)
    # main(cfg)
    # plot_heatmaps(cfg)
    # plot_heatmaps(cfg,plot_index=0)
    # heatmap2_mean_decrease_maskTransfer_exp(cfg)
    # heatmap2_mean_decrease_maskTransfer_exp_to_stochastic(cfg)

    # heatmap1_exp(cfg)
    # cfg = omegaconf.DictConfig({
    #     "population": 10,
    #     "model": "0",
    #     "architecture": "resnet18",
    #     "noise": "geogaussian",
    #     "sigma": 0.0021419609859022197,
    #     "amount": 0.5,
    #     "batch_size":512,
    #     "num_workers":1,
    #     "use_wandb": False
    # })
    # heatmap2_mean_decrease_maskTransfer_exp_to_stochastic(cfg)
    # heatmap1_exp(cfg)
    # heatmap2_mean_decrease_maskTransfer_exp(cfg)
    # plot_heatmaps(cfg)
    # plot_heatmaps(cfg, plot_index=0)
    # main(cfg)
    # noise_calibration(cfg)
    # plot_layer_experiments("a")
    # plot_layer_experiments("b")
    # run_layer_experiment(cfg)
