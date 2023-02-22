import pickle
import glob
# import pygmo
import typing
from typing import List, Union, Any
import pandas as pd
import datetime as date
import wandb
import optuna
# sys.path.append('csgmcmc')
from alternate_models import ResNet, VGG
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
from functools import partial
import glob
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import array as pyarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
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
from matplotlib.axes import SubplotBase
from itertools import chain, combinations
import seaborn as sns
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from plot_utils import plot_ridge_plot, plot_double_barplot, plot_histograms_per_group, stacked_barplot, \
    stacked_barplot_with_third_subplot, plot_double_barplot
from sparse_ensemble_utils import erdos_renyi_per_layer_pruning_rate
from sparse_ensemble_utils import erdos_renyi_per_layer_pruning_rate, get_layer_dict, is_prunable_module, \
    count_parameters, sparsity, get_percentile_per_layer, get_sampler, test, restricted_fine_tune_measure_flops, \
    get_random_batch, efficient_population_evaluation, get_random_image_label, check_for_layers_collapse,get_mask,apply_mask
from itertools import cycle
from matplotlib.patches import PathPatch
# import pylustrator
from shrinkbench.metrics.flops import flops
from pathlib import Path
# enable cuda devices
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# matplotlib.use('TkAgg')


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


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


def save_onnx(cfg):
    data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
                                                                               "University of Leeds\PhD\Datasets\CIFAR10"
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


######################### Noise adding functions #################################################################
def add_geometric_gaussian_noise_to_weights(m, sigma=0.2):
    with torch.no_grad():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            m.weight.multiply_(torch.normal(mean=torch.ones_like(m.weight), std=sigma).to(m.weight.device))


def add_gaussian_noise_to_weights(m, sigma=0.01, adaptive=False):
    with torch.no_grad():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            if adaptive:
                sigma_adaptive = torch.quantile(torch.abs(m.weight), torch.tensor([0.50]))
                m.weight.add_(torch.normal(mean=torch.zeros_like(m.weight), std=sigma_adaptive).to(m.weight.device))
            else:
                m.weight.add_(torch.normal(mean=torch.zeros_like(m.weight), std=sigma).to(m.weight.device))


def add_geogaussian_noise_to_layers(model: torch.nn.Module, sigma_per_layer: dict, exclude_layers: list = []):
    named_modules = model.named_modules()

    with torch.no_grad():
        for name, m in named_modules:
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m,
                                                                                     nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layers:
                sigma = sigma_per_layer[name]
                m.weight.multiply_(torch.normal(mean=torch.ones_like(m.weight), std=sigma).to(m.weight.device))


def add_gaussian_noise_to_layers(model: torch.nn.Module, sigma_per_layer: dict, iterative: bool = False, exclude_layers:
list =
[]):
    named_modules = model.named_modules()
    with torch.no_grad():
        for name, m in named_modules:
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m,
                                                                                     nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layers:
                sigma = sigma_per_layer[name]
                if "weight_mask" in dict(m.named_buffers()).keys() and iterative:
                    weight_mask = dict(m.named_buffers())["weight_mask"]
                    noise = torch.normal(mean=torch.zeros_like(m.weight), std=sigma).to(m.weight.device)
                    noise.mul_(weight_mask)
                    m.weight.data.add_(noise)
                else:
                    m.weight.data.add_(torch.normal(mean=torch.zeros_like(m.weight), std=sigma).to(m.weight.device))

def get_noisy_sample_sigma_per_layer(net: torch.nn.Module, cfg: omegaconf.DictConfig, sigma_per_layer,clone=True):
    current_model = None
    if clone:
        current_model = copy.deepcopy(net)
    else:
        current_model = net
    if cfg.noise == "gaussian":
        add_gaussian_noise_to_layers(current_model, sigma_per_layer=sigma_per_layer, exclude_layers=cfg.exclude_layers)
    elif cfg.noise == "geogaussian":
        add_geogaussian_noise_to_layers(current_model, sigma_per_layer=sigma_per_layer,
                                        exclude_layers=cfg.exclude_layers)
    return current_model


# def get_noisy_sample_pruned_net_work(net: torch.nn.Module, cfg: omegaconf.DictConfig, sigma_per_layer):
def get_noisy_sample(net: torch.nn.Module, cfg: omegaconf.DictConfig):
    current_model = copy.deepcopy(net)
    if cfg.noise == "gaussian":
        current_model.apply(partial(add_gaussian_noise_to_weights, sigma=cfg.sigma))
    elif cfg.noise == "geogaussian":
        current_model.apply(partial(add_geometric_gaussian_noise_to_weights, sigma=cfg.sigma))
    return current_model


##################################################################################################################


def weights_to_prune(model: torch.nn.Module, exclude_layer_list=[]):
    modules = []
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layer_list:
            modules.append((m, "weight"))
    return modules


def get_weights_of_layer_name(model, layer_name):
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            if name == layer_name:
                return (m, "weight")


def remove_reparametrization(model, name_module="", exclude_layer_list: list = []):
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layer_list:
            if name_module == "":
                prune.remove(m, "weight")
            if name == name_module:
                prune.remove(m, "weight")
                break


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
        weights_to_prune = weights_to_prune(pruned_original)
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

                    prune.global_unstructured(weights_to_prune(current_model),
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
                    isinstance(m, nn.BatchNorm3d) and "weight_mask" in list(dict(from_net.named_modules())[
                                                                                name].named_buffers()):
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
        weights_to_prune = weights_to_prune(pruned_original)
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
                    weights_to_prune(current_model),
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
                    weights_to_prune(current_model),
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
    ####################################  Mean decrease heatmap #################################
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
        weights_to_prune = weights_to_prune(pruned_original)
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
                    weights_to_prune(current_model),
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
    string = "{:10.1f}%".format(value).replace(" ", "")
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
    net = get_model(cfg)
    # sigma_add = trial.suggest_float("sigma_add", 0.0001, 0.1)
    sigma_add = cfg.sigma
    sigma_mul = trial.suggest_float("sigma_mul", 0.1, 1)

    # Prune original
    pruned_original = copy.deepcopy(net)
    weights_prune = weights_to_prune(pruned_original)
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
        weights_prune = weights_to_prune(add_noise_model)
        prune.global_unstructured(
            weights_prune,
            pruning_method=prune.L1Unstructured,
            amount=cfg.amount
        )
        remove_reparametrization(model=add_noise_model)
        # Geometric
        weights_prune = weights_to_prune(mul_noise_model)
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


def prune_with_rate(net: torch.nn.Module, amount: typing.Union[int, float], pruner: str = "erk",
                    type: str = "global",
                    criterion:
                    str =
                    "l1", exclude_layers: list = [], pr_per_layer: dict = {}, return_pr_per_layer: bool = False):
    if type == "global":
        weights = weights_to_prune(net, exclude_layer_list=exclude_layers)
        if criterion == "l1":
            prune.global_unstructured(
                weights,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        if criterion == "l2":
            prune.global_unstructured(
                weights,
                pruning_method=prune.LnStructured,
                amount=amount,
                n=2
            )
    elif type == "layer-wise":
        from layer_adaptive_sparsity.tools.pruners import get_modules, get_weights, weight_pruner_loader
        if pruner == "lamp":
            pruner = weight_pruner_loader(pruner)
            if return_pr_per_layer:
                return pruner(model=net, amount=amount, exclude_layers=exclude_layers,
                              return_amounts=return_pr_per_layer)
            else:
                pruner(model=net, amount=amount, exclude_layers=exclude_layers)
        if pruner == "erk":
            pruner = weight_pruner_loader(pruner)
            pruner(model=net, amount=amount, exclude_layers=exclude_layers)
            # _, amount_per_layer, _, _ = erdos_renyi_per_layer_pruning_rate(model=net, cfg=cfg)
            # names, weights = zip(*get_layer_dict(net))
            # for name, module in net.named_modules():
            #     if name in exclude_layers or name not in names:
            #         continue
            #     else:
            #         prune.l1_unstructured(module, name="weight", amount=float(amount_per_layer[name]))
        if pruner == "manual":
            for name, module in net.named_modules():
                with torch.no_grad():
                    if name in exclude_layers or not is_prunable_module(module):
                        continue
                    else:
                        prune.l1_unstructured(module, name="weight", amount=float(pr_per_layer[name]))



    else:
        raise NotImplementedError("Not implemented for type {}".format(type))


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
    if cfg.dataset == "cifar10":
        # data_path = "/nobackup/sclaam/data" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
        #
        #                                                                            "University of Leeds\PhD\Datasets\CIFAR10"
        current_directory = Path().cwd()
        data_path = ""
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path ="/nobackup/sclaam/data"
        elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
            data_path ="C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"
        # data_path = "datasets" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
        #                                                                            "University of Leeds\PhD\Datasets\CIFAR10"

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
    if cfg.dataset == "cifar100":
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

        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        cifar10_train, cifar10_val = random_split(trainset, [45000, 5000])

        trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=cfg.batch_size, shuffle=True,
                                                  num_workers=cfg.num_workers)
        val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=cfg.batch_size, shuffle=True,
                                                 num_workers=cfg.num_workers)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
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
    weights_to_prune = weights_to_prune(pruned_original)
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
            weights_to_prune(current_model),
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


def one_shot_iterative_sotchastic_pruning(cfg):
    # TODO implement the algorithm for one shot pruning wit optimization of the pruning rate and the
    #

    pass


###############################  Channel inspection
# ############################################
def channel_inspection_of_convolutional_layers(pruned_model: torch.nn.Module, exclude_layer_list=[]):
    histograms_of_0_per_layer = {}
    for name, m in pruned_model.named_modules():
        if hasattr(m, 'weight') and isinstance(m, nn.Conv2d) and name not in exclude_layer_list:

            # input channels X output shanels X dim1 kernel X dim2 kernel
            weight_copy = m.weight.detach().clone().numpy()
            in_channels, out_channels, dim1k, dim2k = weight_copy.shape

            for output_index in range(out_channels):
                slice_weight = weight_copy[:, output_index, :, :]
                slice_weight.reshape((in_channels, dim1k * dim2k))

            # histograms_of_0_per_layer[]

            modules.append((m, "weight"))


############################### Experiments 25 of October # ############################################################

def select_eval_set(cfg: omegaconf.DictConfig, eval_set: str):
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    evaluation_set = None
    if eval_set == "test":
        evaluation_set = testloader
    elif eval_set == "val":
        evaluation_set = valloader
    elif eval_set == "train":
        evaluation_set = trainloader
    else:
        raise Exception("Invalid evaluation set {} is not valid".format(eval_set))
    return evaluation_set


def transfer_mask_rank_experiments(cfg: omegaconf.DictConfig, eval_set: str = "test"):
    use_cuda = torch.cuda.is_available()
    net = get_model(cfg)
    evaluation_set = select_eval_set(cfg, eval_set)
    N = cfg.population
    pop = []
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []
    deterministic_with_stochastic_mask_performance = []
    stochastic_with_deterministic_mask_performance = []
    original_performance = test(net, use_cuda, evaluation_set, verbose=1)
    pruned_original = copy.deepcopy(net)
    prune_with_rate(pruned_original, cfg.amount)
    # remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(pruned_original, use_cuda, evaluation_set, verbose=1)
    pop.append(pruned_original)
    pruned_performance.append(pruned_original_performance)
    labels = ["pruned original"]
    stochastic_dense_performances.append(original_performance)
    for n in range(N):
        sto_mask_transfer_model = copy.deepcopy(net)

        current_model = get_noisy_sample(net, cfg)

        det_mask_transfer_model = copy.deepcopy(current_model)
        copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)

        stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        print("Stochastic dense performance")
        StoDense_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        # Dense stochastic performance
        stochastic_dense_performances.append(StoDense_performance)

        prune_with_rate(current_model, amount=cfg.amount)

        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model)
        print("Performance for model {}".format(n))

        stochastic_pruned_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        deterministic_with_stochastic_mask_performance.append(test(sto_mask_transfer_model, use_cuda, evaluation_set,
                                                                   verbose=1))
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
    # np.save(
    #     "data/population_data/performances_{}_transfer_t_{}-{}.npy".format(cfg.noise, result.tm_hour, result.tm_min),
    #     pruned_performance)
    # for i, model in enumerate(pop):
    #     with open("data/population_data/{}/model_{}.pkl".format(cfg.noise, i), "wb") as f:
    #         pickle.dump(model, f)
    del pop
    cutoff = original_performance - 2
    # plot_population_ranking_with_cutoff(cfg, original_performance, cutoff, pruned_performance, ranked_index,
    #                                     stochastic_dense_performances, index_of_pruned_original, labels, result,
    #                                     eval_set, "static")
    ################################# plotting the comparison #########################################################
    fig = plt.figure()
    original_line = plt.axhline(y=original_performance, color="k", linestyle="-", label="Original Performance")
    # plt.axhline(y=cutoff, color="r", linestyle="--", label="cutoff value")
    plt.xlabel("Ranking index", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    sigma = format_sigmas(cfg.sigma)
    pr = format_percentages(cfg.amount)
    # if cfg.noise == "geogaussian":
    #     plt.title(f"CIFAR10 Geometric Gaussian Noise $\sigma={sigma},pr={pr}$", fontsize=10)
    # if cfg.noise == "gaussian":
    #     plt.title(f"CIFAR10 Additive Gaussian Noise $\sigma={sigma},pr={pr}$", fontsize=10)

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

    plt.legend([original_line, tuple(stochastic_models_points_pruned), tuple(stochastic_models_points_dense), p1,
                tuple(transfer_mask_models_points), tuple(stochastic_with_deterministic_mask_models_points)],
               ["Original Performance", 'Pruned Stochastic', 'Dense Stochastic', "Original Accuracy", "Stochastic Mask "
                                                                                                      "" + r"$\rightarrow$"
                + " Original Weights", "Deterministic Mask" + r"$\rightarrow$" + "Stochastic Weights"],
               scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)})
    plt.tight_layout()
    plt.savefig(
        f"data/figures/transfers_comparison_{cfg.noise}_sigma_"
        f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
        f"_{cfg.population}"
        f"_t_{result.tm_hour}"
        f"-{result.tm_min}_{eval_set}.pdf")
    plt.savefig(f"data/figures/transfers_comparison_{cfg.noise}_sigma_{cfg.sigma}_pr_{cfg.amount}_batchSize"
                f"_{cfg.batch_size}_pop"
                f"_{cfg.population}"
                f"_t_{result.tm_hour}"
                f"-{result.tm_min}_{eval_set}.png")


def transfer_mask_rank_experiments_plot_adaptive_noise(cfg: omegaconf.DictConfig, eval_set: str = "test"):
    use_cuda = torch.cuda.is_available()
    net = get_model(cfg)
    quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
                                     names=["layer", "q25", "q50", "q75"])

    sigma_per_layer = quantile_per_layer.set_index('layer')["q25"].T.to_dict()

    evaluation_set = select_eval_set(cfg, eval_set)

    N = cfg.population
    pruned_performance = []
    stochastic_static_dense_performances = []
    stochastic_dynamic_dense_performances = []

    dynamic_pruned_performance = []
    stochastic_deltas = []
    sto_mask_to_ori_weights_deltas = []
    ori_mask_to_sto_weights_deltas = []

    original_with_stochastic_mask_performance = []
    stochastic_with_deterministic_mask_performance = []
    original_performance = test(net, use_cuda, evaluation_set)
    pruned_original = copy.deepcopy(net)

    prune_with_rate(pruned_original, amount=cfg.amount, exclude_layers=cfg.exclude_layers)

    # remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(pruned_original, use_cuda, evaluation_set, verbose=1)

    pruned_performance.append(pruned_original_performance)
    stochastic_static_dense_performances.append(original_performance)

    plot_labels = ["pruned original"]

    for n in range(N):
        sto_mask_transfer_model = copy.deepcopy(net)
        current_model_static_noise = get_noisy_sample(net, cfg)

        stochastic_static_dense_performance = test(current_model_static_noise, use_cuda, evaluation_set, verbose=1)
        stochastic_static_dense_performances.append(stochastic_static_dense_performance)
        current_model_dynamic_noise = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        stochastic_dynamic_dense_performance = test(current_model_static_noise, use_cuda, evaluation_set, verbose=1)
        stochastic_dynamic_dense_performances.append(stochastic_dynamic_dense_performance)

        det_mask_transfer_model = copy.deepcopy(current_model_static_noise)
        copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        # det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)
        # print("Performance for dense model {}".format(n))
        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)

        # prune current model static noise
        prune_with_rate(current_model_static_noise, cfg.amount, exclude_layers=cfg.exclude_layers)

        # prune current model dynamic noise

        prune_with_rate(current_model_dynamic_noise, cfg.amount, exclude_layers=cfg.exclude_layers)

        # Here is where I transfer the mask from the prune stochastic model to the
        # original weights and put it in the ranking

        copy_buffers(from_net=current_model_static_noise, to_net=sto_mask_transfer_model)

        # remove_reparametrization(current_model_static_noise)

        print("Performance for static stochastic pruned model {}".format(n))
        stochastic_pruned_performance = test(current_model_static_noise, use_cuda, evaluation_set, verbose=1)
        print("Performance for dynamic stochastic pruned model {}".format(n))
        stochastic_dynamic_pruned_performance = test(current_model_dynamic_noise, use_cuda, evaluation_set, verbose=1)

        # print("Performance for Transfer sto mask {} to model".format(n))
        # # original_with_sto_mask = test(sto_mask_transfer_model, use_cuda, evaluation_set, verbose=1)
        # original_with_stochastic_mask_performance.append(original_with_sto_mask)

        pruned_performance.append(stochastic_pruned_performance)
        dynamic_pruned_performance.append(stochastic_dynamic_pruned_performance)

        stochastic_deltas.append(original_performance - stochastic_pruned_performance)
        # sto_mask_to_ori_weights_deltas.append(original_performance - original_with_sto_mask)
        # ori_mask_to_sto_weights_deltas.append(original_performance - det_mask_transfer_model_performance)
    # len(pruned performance)-1 because the first one is the pruned original
    # Deterministic mask transfer to stochastic model
    plot_labels.extend(["static stochastic pruned"] * (len(pruned_performance) - 1))
    plot_labels.extend(["dynamic stochastic pruned"] * (len(dynamic_pruned_performance)))
    pruned_performance.extend(dynamic_pruned_performance)
    stochastic_static_dense_performances.extend(stochastic_dynamic_dense_performances)

    # Deterministic mask transfer to stochastic model
    # plot_labels.extend(["det mask transfer"] * len(stochastic_with_deterministic_mask_performance))
    # pruned_performance.extend(stochastic_with_deterministic_mask_performance)
    # stochastic_static_dense_performances.extend([1] * len(stochastic_with_deterministic_mask_performance))

    # This gives a list of the INDEXES that would sort "pruned_performance". I know that the index 0 of
    # pruned_performance is the pruned original. Then I ask ranked index where is the element 0 which references the
    # index 0 of pruned_performance.

    assert len(plot_labels) == len(pruned_performance), f"The labels and the performances are not the same length: " \
                                                        f"{len(plot_labels)}!={len(pruned_performance)}"

    ranked_index = np.flip(np.argsort(pruned_performance))
    index_of_pruned_original = list(ranked_index).index(0)

    pruned_performance = np.array(pruned_performance)
    stochastic_static_dense_performances = np.array(stochastic_static_dense_performances)

    result = time.localtime(time.time())
    epsilon = []
    labels = []
    epsilon.extend(stochastic_deltas)
    labels.extend(["stochastic pruning"] * len(stochastic_deltas))
    epsilon.extend(sto_mask_to_ori_weights_deltas)
    labels.extend(["Sto. mask original weights"] * len(sto_mask_to_ori_weights_deltas))
    epsilon.extend(ori_mask_to_sto_weights_deltas)
    labels.extend(["Det. mask sto. weights"] * len(ori_mask_to_sto_weights_deltas))
    df_sigmas = [format_sigmas(cfg.sigma)] * len(labels)
    df_N = [cfg.population] * len(labels)
    df_pr = [cfg.amount] * len(labels)
    results = {"Epsilon": epsilon, "Population": df_N, "Type": labels, "Pruning Rate": df_pr, "sigma": df_sigmas}
    df = pd.DataFrame(results)

    # plot_population_ranking_with_cutoff(cfg, original_performance, cutoff=original_performance - 2,
    #                                     pruned_performance=pruned_performance, ranked_index=ranked_index,
    #                                     stochastic_static_dense_performances=stochastic_static_dense_performances,
    #                                     index_of_pruned_original=index_of_pruned_original,
    #                                     labels=labels, time_object=result, eval_set=eval_set, identifier="ADAPTIVE")
    cutoff = original_performance - 2
    ######################################### Plot the comparisons #####################################################
    fig = plt.figure()
    plt.axhline(y=original_performance, color="k", linestyle="-", label="Original dense pruned_performance")
    plt.axhline(y=cutoff, color="r", linestyle="--", label="cutoff value")
    plt.xlabel("Ranking index", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    sigma = format_sigmas(cfg.sigma)
    pr = format_percentages(cfg.amount)
    if cfg.noise == "geogaussian":
        plt.title(f"CIFAR10 Geometric Gaussian Noise $pr={pr}$", fontsize=10)
    if cfg.noise == "gaussian":
        plt.title(f"CIFAR10 Additive Gaussian Noise $pr={pr}$", fontsize=10)

    stochastic_static_models_points_dense = []
    stochastic_static_models_points_pruned = []

    stochastic_dynamic_models_points_dense = []
    stochastic_dynamic_models_points_pruned = []

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
            if plot_labels[ranked_index[i]] == "sto mask transfer":
                sto_transfer_point = plt.scatter(i, element, c="tab:orange", marker="P")
                transfer_mask_models_points.append(sto_transfer_point)
            elif plot_labels[ranked_index[i]] == "det mask transfer":
                det_transfer_point = plt.scatter(i, element, c="tab:olive", marker="X")
                stochastic_with_deterministic_mask_models_points.append(det_transfer_point)

            elif plot_labels[ranked_index[i]] == "static stochastic pruned":
                pruned_point = plt.scatter(i, element, c="b", marker="x")
                stochastic_static_models_points_pruned.append(pruned_point)
            elif plot_labels[ranked_index[i]] == "dynamic stochastic pruned":
                pruned_point = plt.scatter(i, element, c="b", marker="d")
                stochastic_dynamic_models_points_pruned.append(pruned_point)

    for i, element in enumerate(stochastic_static_dense_performances[ranked_index]):
        if i == index_of_pruned_original or element == 1:
            continue
            # plt.scatter(i, element, c="y", marker="o", label="original model performance")
        elif plot_labels[ranked_index[i]] == "static stochastic pruned":
            dense_point = plt.scatter(i, element, c="c", marker="1")
            stochastic_static_models_points_dense.append(dense_point)
        elif plot_labels[ranked_index[i]] == "dynamic stochastic pruned":
            dense_point = plt.scatter(i, element, c="c", marker="D")
            stochastic_dynamic_models_points_dense.append(dense_point)

    plt.legend([tuple(stochastic_static_models_points_pruned), tuple(stochastic_static_models_points_dense), p1,
                tuple(stochastic_dynamic_models_points_pruned), tuple(stochastic_dynamic_models_points_dense)],
               ['Pruned static stochastic', 'Dense static stochastic', "Original model pruned",
                "Pruned dynamic stochastic",
                "Dense dynamic stochastic"],
               scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)})
    plt.tight_layout()
    plt.savefig(
        f"data/figures/adaptive_V_static_comparison_{cfg.noise}_sigma_"
        f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
        f"_{cfg.population}"
        f"_t_{result.tm_hour}"
        f"-{result.tm_min}_{eval_set}.pdf")
    plt.savefig(f"data/figures/adaptive_V_static_comparison_{cfg.noise}_sigma_{cfg.sigma}_pr_{cfg.amount}_batchSize"
                f"_{cfg.batch_size}_pop"
                f"_{cfg.population}"
                f"_t_{result.tm_hour}"
                f"-{result.tm_min}_{eval_set}.png")
    return df


def transfer_mask_rank_experiments_no_plot(cfg: omegaconf.DictConfig):
    use_cuda = torch.cuda.is_available()
    net = get_model(cfg)
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    N = cfg.population
    pruned_performance = []
    stochastic_deltas = []
    sto_mask_to_ori_weights_deltas = []
    ori_mask_to_sto_weights_deltas = []

    original_with_stochastic_mask_performance = []
    stochastic_with_deterministic_mask_performance = []

    original_performance = test(net, use_cuda, testloader)
    pruned_original = copy.deepcopy(net)
    weights_to_prune = weights_to_prune(pruned_original)
    prune.global_unstructured(
        weights_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=cfg.amount
    )
    # remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(pruned_original, use_cuda, testloader)
    pruned_performance.append(pruned_original_performance)
    for n in range(N):
        sto_mask_transfer_model = copy.deepcopy(net)
        current_model = get_noisy_sample(net, cfg)
        det_mask_transfer_model = copy.deepcopy(current_model)
        copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, testloader)
        stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        prune.global_unstructured(
            weights_to_prune(current_model),
            pruning_method=prune.L1Unstructured,
            amount=cfg.amount)
        # Here is where I transfer the mask from the prune stochastic model to the
        # original weights and put it in the ranking

        copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model)
        print("Performance for model {}".format(n))
        stochastic_pruned_performance = test(current_model, use_cuda, testloader)
        original_with_sto_mask = test(sto_mask_transfer_model, use_cuda, testloader)
        original_with_stochastic_mask_performance.append(original_with_sto_mask)
        pruned_performance.append(stochastic_pruned_performance)
        stochastic_deltas.append(original_performance - stochastic_pruned_performance)
        sto_mask_to_ori_weights_deltas.append(original_performance - original_with_sto_mask)
        ori_mask_to_sto_weights_deltas.append(original_performance - det_mask_transfer_model_performance)
    # len(pruned performance)-1 because the first one is the pruned original
    # Deterministic mask transfer to stochastic model

    # This gives a list of the INDEXES that would sort "pruned_performance". I know that the index 0 of
    # pruned_performance is the pruned original. Then I ask ranked index where is the element 0 which references the
    # index 0 of pruned_performance.
    result = time.localtime(time.time())
    epsilon = []
    labels = []
    epsilon.extend(stochastic_deltas)
    labels.extend(["stochastic pruning"] * len(stochastic_deltas))
    epsilon.extend(sto_mask_to_ori_weights_deltas)
    labels.extend(["Sto. mask original weights"] * len(sto_mask_to_ori_weights_deltas))
    epsilon.extend(ori_mask_to_sto_weights_deltas)
    labels.extend(["Det. mask sto. weights"] * len(ori_mask_to_sto_weights_deltas))
    df_sigmas = [format_sigmas(cfg.sigma)] * len(labels)
    df_N = [cfg.population] * len(labels)
    df_pr = [cfg.amount] * len(labels)
    results = {"Epsilon": epsilon, "Population": df_N, "Type": labels, "Pruning Rate": df_pr, "sigma": df_sigmas}
    df = pd.DataFrame(results)
    # df = df.append(results,ignore_index=True)
    return df


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
    pruning_rates = [0.5, 0.6, 0.8, 0.83, 0.85, 0.87, 0.9, 0.95]
    for sig in sigmas:
        for pr in pruning_rates:
            cfg.sigma = float(sig)
            cfg.amount = pr

            transfer_mask_rank_experiments(cfg)


def single_statistics_of_epsilon_for_stochastic_pruning(filepath):
    df = pd.read_csv(filepath_or_buffer=filepath, sep=",", header=0)
    # num_col = len(df["Pruning Rate"].unique())
    # num_row = len(df[sigma"].unique())
    # num_col = 3
    # num_row = 3
    # fig = plt.figure(figsize=(16, 10))
    # subfigs = fig.subfigures(num_row, 1, wspace=0.07)
    grped_bplot = sns.catplot(x='Population',
                              y='Epsilon',
                              hue="Type",
                              kind="box",
                              legend=False,
                              height=6,
                              aspect=1.3,
                              data=df)  # .set(title="Title")
    # make grouped stripplot
    grped_bplot = sns.stripplot(x='Population',
                                y='Epsilon',
                                hue='Type',
                                jitter=True,
                                dodge=True,
                                marker='o',
                                edgecolor="gray",
                                linewidth=1,
                                # palette="set2",
                                alpha=0.5,
                                data=df)
    # how to remove redundant legends in Python
    # Let us first get legend information from the plot object
    handles, labels = grped_bplot.get_legend_handles_labels()
    # grped_bplot.set_title("The title")
    plt.title("Pruning Rate 0.9" + r" $\sigma=0.001$", fontsize=12)
    # specify just one legend
    l = plt.legend(handles[:3], labels[:3])
    plt.savefig("grouped_boxplot_with_jittered_data_points_Seaborn_Python2.png", bbox_inches="tight")


def get_model(cfg: omegaconf.DictConfig):
    net = None
    if cfg.architecture == "resnet18":
        if not cfg.solution:
            if "csgmcmc" == cfg.type:
                net = ResNet18()
                return net
            if "alternative" == cfg.model_type:
                from alternate_models.resnet import ResNet18
                net = ResNet18()
                return net
        else:
            if "csgmcmc" == cfg.model_type:
                net = ResNet18()
            if "alternative" == cfg.model_type:
                from alternate_models.resnet import ResNet18
                net = ResNet18()
            load_model(net, cfg.solution)
            return net
    if cfg.architecture == "VGG19":
        if not cfg.solution:
            if "csgmcmc" == cfg.model_type:
                net = VGG(cfg.architecture)
                return net
            if "alternative" == cfg.model_type:
                from alternate_models.vgg import VGG
                net = VGG(cfg.architecture)
                return net
        else:
            if "csgmcmc" == cfg.model_type:
                net = VGG(cfg.architecture)
            if "alternative" == cfg.model_type:
                from alternate_models.vgg import VGG
                net = VGG(cfg.architecture)
            load_model(net, cfg.solution)
            return net

    else:
        raise NotImplementedError("Not implemented for architecture:{}".format(cfg.architecture))


######################################## Check functions ##########################################################
def layers_to_vector(layers_params: typing.List[torch.Tensor]):
    flatten_list = [tensor.flatten().detach() for tensor in layers_params]
    full_vector = torch.cat(flatten_list)
    return full_vector


def get_threshold_and_pruned_vector_from_pruning_rate(list_of_layers: typing.List[torch.Tensor], pruning_rate: float):
    full_vector = layers_to_vector(list_of_layers)
    magnitudes = torch.abs(full_vector)
    total = len(full_vector)
    sorted_index = torch.argsort(magnitudes)
    index_of_threshold = math.floor(total * pruning_rate)
    threshold = magnitudes[sorted_index[index_of_threshold]]
    full_vector[sorted_index[:index_of_threshold]] = 0
    number_of_zeros = len(torch.where(full_vector == 0)[0])
    assert number_of_zeros == index_of_threshold, "Length of 0s in full vector is not the same as the" \
                                                  f"index threshold: " \
                                                  f"{number_of_zeros}!=" \
                                                  f"{index_of_threshold}"

    return float(threshold), index_of_threshold, full_vector


def get_df_pruned_by_layer(names: typing.List[str], weights: typing.List[torch.nn.Module], function: typing.Callable,
                           cfg: omegaconf.DictConfig) -> typing.Tuple[pd.DataFrame, float, int, torch.Tensor]:
    elements = np.array(list(map(function, weights)))
    sorted_indexes_by_function = np.flip(np.argsort(elements))

    threshold, index_of_threshold, pruned_vector = get_threshold_and_pruned_vector_from_pruning_rate(weights,
                                                                                                     cfg.amount)
    new_weights = copy.deepcopy(weights)

    vector_to_parameters(pruned_vector, new_weights)

    layer_survived = []
    layer_died = []
    for w in [new_weights[i] for i in sorted_indexes_by_function]:
        survived = len(torch.where(w != 0)[0])
        died = w.nelement() - survived
        layer_survived.append(survived)
        layer_died.append(died)
    le_dictionary = {"Layer": [names[i] for i in sorted_indexes_by_function], "Survived": layer_survived,
                     "Pruned": layer_died}
    df = pd.DataFrame(le_dictionary)
    return df, threshold, index_of_threshold, pruned_vector


def get_df_changes_by_layer(names: typing.List[str], weights: typing.List[torch.nn.Module], vector1: torch.Tensor,
                            vector2: torch.Tensor,
                            function:
                            typing.Callable = None):
    if function:
        elements = np.array(list(map(function, weights)))
        sorted_indexes_by_function = np.flip(np.argsort(elements))

    difference_vector = torch.bitwise_xor(vector1.type(torch.bool), vector2.type(torch.bool))
    w = copy.deepcopy(weights)
    vector_to_parameters(difference_vector, w)
    number_different_per_layer = []
    #
    for diff_vec in w:
        number_different_per_layer.append(diff_vec.sum().item())
    if function:
        le_dictionary = {"Layer": [names[i] for i in sorted_indexes_by_function],
                         "Changed": [number_different_per_layer[i]
                                     for i in sorted_indexes_by_function]}
    else:
        le_dictionary = {"Layer": names, "Changed": number_different_per_layer}

    df = pd.DataFrame(le_dictionary)
    return df, w


def check_noise_impact_in_weights_by_layer(cfg):
    net = get_model(cfg)
    names, weights = zip(*get_layer_dict(net))
    N = cfg.population
    # BarPlot the original weights pruned at level cfg.amount
    number_param = lambda w: w.nelement()
    df, threshold, index_threshold, pruned_vector = get_df_pruned_by_layer(names=names, weights=weights,
                                                                           function=number_param, cfg=cfg)
    pruned_original = copy.deepcopy(net)
    prune_with_rate(pruned_original, amount=cfg.amount)
    deterministic_pruned_buffer_vector = parameters_to_vector(pruned_original.buffers())
    names_pruned, weights_pruned = zip(*get_layer_dict(pruned_original))

    # remove_reparametrization(pruned_original)
    # zeros = count_zero_parameters(pruned_original)
    # stacked_barplot(df,
    #                 x="Layer",
    #                 y1="Survived",
    #                 y2="Pruned",
    #                 ylabel="Count",
    #                 path="test_pr_{}.pdf".format(cfg.amount),
    #                 title="Pruning rate " f"{cfg.amount} " + f" Pruning Threshold={threshold}",
    #                 label1="Survived",
    #                 label2="Pruned",
    #                 rot=90,
    #                 logscale=True
    #                 )
    # plt.close()

    first_sample = get_noisy_sample(net, cfg)
    prune_with_rate(first_sample, amount=cfg.amount)
    stochastic_pruned_buffer_vector = parameters_to_vector(first_sample.buffers())
    # remove_reparametrization(first_sample)
    df_stochastic, changes_bool_vector_list = get_df_changes_by_layer(names, weights,
                                                                      deterministic_pruned_buffer_vector,
                                                                      stochastic_pruned_buffer_vector,
                                                                      number_param)
    elements = np.array(list(map(number_param, weights)))
    sorted_indexes_by_function = np.flip(np.argsort(elements))
    elements_sorted = elements[sorted_indexes_by_function]
    df_stochastic["Changed Percentage"] = df_stochastic["Changed"] / elements_sorted
    df_stochastic["Total Parameters"] = elements_sorted
    df_stochastic.to_csv("data/analysis_per_layer/weights_altered_per_layer_one_sample_pr_{}.csv".format(cfg.amount),
                         sep=",", index=False)
    fig = plt.figure()
    ax = df_stochastic.plot.bar(x='Layer', y='Changed Percentage', rot=90)
    ax.set_xlabel("Layers by number of parameters " + r"$+\;\rightarrow\;-$")
    # ax.set_yscale("log")
    plt.ylabel("Percentage")
    plt.title("Pruning rate = {}".format(cfg.amount))
    plt.tight_layout()
    plt.savefig(f"data/analysis_per_layer/changes_percentages_single_sample_pr_{cfg.amount}.pdf")
    plt.savefig(f"data/analysis_per_layer/changes_percentages_single_sample_pr_{cfg.amount}.png")
    plt.close()
    fig = plt.figure()
    ax = df_stochastic.plot.bar(x='Layer', y='Changed', rot=90)
    ax.set_xlabel("Layers by number of parameters " + r"$+\;\rightarrow\;-$")
    ax.set_yscale("log")
    plt.ylabel("Count")
    plt.title("Pruning rate = {}".format(cfg.amount) + "Static sigma ={}".format(cfg.sigma))
    plt.tight_layout()
    plt.savefig(f"data/analysis_per_layer/changes_single_sample_pr_{cfg.amount}_sigma_{cfg.sigma}.pdf")
    plt.savefig(f"data/analysis_per_layer/changes_single_sample_pr_{cfg.amount}_sigma_{cfg.sigma}.png")
    plt.close()
    return threshold


def heatmap_scaled_noise_per_layer(cfg: omegaconf.DictConfig):
    """
    This is for a experiment that uses stochastic noise only on
    one layer at the time and plots a heatmap for different pruning rates.
    :param cfg:
    :return:
    """
    pass


def iterative_stochastic_pruning(cfg: omegaconf.DictConfig):
    pass


def two_step_iterative_pruning(cfg: omegaconf.DictConfig):
    pass


def gradient_decent_on_sigma_pr():
    pass


def weights_analysis_per_weight(cfg: omegaconf.DictConfig):
    net = get_model(cfg)
    names, weights = zip(*get_layer_dict(net))
    average_magnitude = lambda w: torch.abs(w).mean()
    average_magnitudes_by_layer = np.array(list(map(average_magnitude, weights)))
    number_param = lambda w: w.nelement()
    elements = np.array(list(map(number_param, weights)))
    ratios = average_magnitudes_by_layer / cfg.sigma
    sorted_idexes_by_ratios = np.flip(np.argsort(ratios))
    # weights_magnitude_by_size = [np.abs(weights[i].flatten().detach().numpy()) for i in sorted_idexes_by_size]
    # names_by_size = [names[i] for i in sorted_idexes_by_size]
    # n = np.array([],dtype=str)
    # we = pyarr.array("f", [])
    # for j, w in enumerate(weights_magnitude_by_size):
    #     we.extend(w)
    #     for i in range(len(w)):
    #        n = np.append(n,names_by_size[j])
    # df = pd.DataFrame(data={"x": float(we), "g": n})
    # df.to_csv(
    #     "data/weights_by_size.csv", sep=",", index=False
    # )
    df = pd.read_csv("data/weights_by_size.csv", header=0, sep=",")
    # # plot_ridge_plot(df, "data/figures/original_weights_ridgeplot.png".format(cfg.sigma))
    df.rename(columns={"g": "Layer Name", "x": "Weight magnitude"}, inplace=True)
    df["Weight magnitude"] = df['Weight magnitude'].apply(lambda x: np.abs(x))
    print(df)

    def q25(x):
        return x.quantile(0.25)

    def q50(x):
        return x.quantile(0.50)

    def q75(x):
        return x.quantile(0.75)

    # vals = {'Weight magnitude': [q25, q50, q75]}
    # quantile_df = df.groupby('Layer Name').agg(vals)
    # quantile_df = df.groupby("").quantile([0.25,0.5,0.75])
    plot_histograms_per_group(df, "Weight magnitude", "Layer Name")
    fancy_bloxplot(df, x="Layer Name", y="Weight magnitude", rot=90)
    # print(quantile_df)
    # quantile_df.to_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", index=True)

    ########################## This is double bar plot ################################################

    # y_axes_label = r"$\frac{\bar{\mid w \mid}}{\sigma}$"
    # title = r"$\sigma = {}$".format(cfg.sigma)
    #
    # df2 = pd.DataFrame(data={"y1": ratios[sorted_idexes_by_ratios].transpose(), "y2": elements[
    #     sorted_idexes_by_ratios].transpose()})
    # xtick_labels = [names[i] for i in sorted_idexes_by_ratios]
    # plot_double_barplot(df2,y_axes_label,"Number of parameters",title,f"data/figures/sigma_"
    #                                                                  f"{cfg.sigma}_V_original_weights.png",
    #                     xtick_labels,logy2=True)


def epsilon_for_pruning_rate_given_best_sigma(filepath: str, cfg: omegaconf.DictConfig):
    df = pd.read_csv(filepath_or_buffer=filepath, sep=",", header=0)
    epsilons = []
    pruning_rates = []
    num_col = len(df["Pruning Rate"].unique())
    num_row = len(df["sigma"].unique())
    minimun_overall = df["Epsilon"].min()
    print("The minimun epsilon is {} with the following charactersitics".format(minimun_overall))
    print(df[df["Epsilon"] == minimun_overall])
    for pr in df["Pruning Rate"].unique():
        current_df = df[df["Pruning Rate"] == pr]
        best_epsilon = current_df.minimun()
        best_epsilon_info = current_df[current_df["Epsilon"] == best_epsilon]
        print("For pruning rate {}  the best epsilon is {} obtained with the following parameters \n {}".format(pr,
                                                                                                                best_epsilon,
                                                                                                                best_epsilon_info))
        epsilons.append(best_epsilon)
        pruning_rates.append(pr)
    plt.plot(epsilons, pruning_rates)
    plt.ylabel(r"$\epsilon\mid_{\sigma_{best}}$")
    plt.xlabel("Pruning rate")
    plt.savefig("data/figures/best_epsilon.png")


def plot_specific_pr_sigma_epsilon_statistics(filepath: str, cfg: omegaconf.DictConfig, specific_sigmas: list,
                                              specific_pruning_rates: list):
    use_cuda = torch.cuda.is_available()

    net = get_model(cfg)

    _, valloader, testloader = get_cifar_datasets(cfg)

    original_performance = test(net, use_cuda, testloader)

    df = pd.read_csv(filepath_or_buffer=filepath, sep=",", header=0)
    num_col = len(specific_pruning_rates)
    # pylustrator.start()
    # num_col = 3
    # num_row = 3
    # For each num_row is going to be one plot with num_col subplots for each pruning rate
    for i, current_sigma in enumerate(specific_sigmas):
        # current_sigma = df["sigma"].unique()[i]
        # if num_col == 1:
        #     fig = plt.figure(figsize=(13, 13 ))
        #     axs_p = fig.subplot()
        fig, axs_p = plt.subplots(1, num_col, figsize=(9
                                                       , 10))
        axs = []
        if isinstance(axs_p, SubplotBase):
            axs.append(axs_p)
        else:
            axs.extend(axs_p)

        # plt.suptitle(r"$\sigma={}$".format(current_sigma), fontsize=20)
        for j, axj in enumerate(axs):
            current_pr = specific_pruning_rates[j]
            # axj = fig.add_subplot(1, num_col, j+1,)
            current_df = df[df["sigma"] == current_sigma]
            current_df = current_df[current_df["Pruning Rate"] == current_pr]
            current_df["Accuracy"] = original_performance - current_df["Epsilon"]
            pruned_original = copy.deepcopy(net)
            prune_with_rate(pruned_original, float(current_pr))
            remove_reparametrization(pruned_original)
            pruned_original_performance = test(pruned_original, use_cuda, testloader, verbose=0)
            delta_pruned_original_performance = original_performance - pruned_original_performance
            ###############  LAMP ################################
            total_observations = len(current_df["Accuracy"][current_df["Type"] == "Stochastic Pruning"])

            lamp_model = copy.deepcopy(net)

            prune_with_rate(lamp_model, float(current_pr), exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner="lamp")
            print("Lamp deterministic performance")
            LAMP_deterministic_performance = test(lamp_model, use_cuda, testloader, verbose=0)
            print(LAMP_deterministic_performance)
            print("Lamp deterministic performance on val set pruning rate {}".format(current_pr))
            LAMP_deterministic_performance_val = test(lamp_model, use_cuda, valloader, verbose=0)
            print(LAMP_deterministic_performance_val)
            print("GLOBAL deterministic performance")
            print(pruned_original_performance)
            print("Global deterministic performance on val set pruning rate {}".format(current_pr))
            GLOBAL_deterministic_performance_val = test(pruned_original, use_cuda, valloader, verbose=0)
            print(GLOBAL_deterministic_performance_val)
            for type in current_df["Type"].unique():
                number_of_elements_above_LAMP_determinstic = (
                        current_df["Accuracy"][type == current_df["Type"]] > LAMP_deterministic_performance).sum()
                number_of_elements_above_GLOBAL_determinstic = (
                        current_df["Accuracy"][type == current_df["Type"]] > pruned_original_performance).sum()
                print(
                    "For pruning rate {} and sigma {} the number of elements above deterministic LAMP for type {} are {} then the fraction is {}".format(
                        current_pr, current_sigma, type, number_of_elements_above_LAMP_determinstic,
                        number_of_elements_above_LAMP_determinstic / total_observations))
                print(
                    "For pruning rate {} and sigma {} the number of elements above deterministic Global for type {} are {} then the fraction is {}".format(
                        current_pr, current_sigma, type, number_of_elements_above_GLOBAL_determinstic,
                        number_of_elements_above_GLOBAL_determinstic / total_observations))

            axj = sns.boxplot(x='Type',
                              y='Accuracy',
                              hue="Type",
                              data=current_df,
                              ax=axj
                              # width = 1
                              )

            adjust_box_widths(fig, 2)
            axj.axhline(pruned_original_performance, c="purple", linewidth=2.5, label="Global Deterministic Pruning")
            axj.axhline(LAMP_deterministic_performance, c="xkcd:greeny yellow", linewidth=2.5, label="LAMP "
                                                                                                     "Deterministic "
                                                                                                     "Pruning")

            axj = sns.stripplot(x='Type',
                                y='Accuracy',
                                hue='Type',
                                jitter=True,
                                dodge=True,
                                marker='o',
                                edgecolor="gray",
                                linewidth=1,
                                # palette="set2",
                                alpha=0.5,
                                data=current_df,
                                ax=axj)
            # axj.set_title("Pruning Rate = {}".format(current_pr), fontsize=15)
            # plt.ylabel("Accuracy", fontsize=20)
            # ticks = g.get_yticks(minor=False)
            # accuracy_ticks = original_performance - ticks
            # axj.set_ylabel("Accuracy", fontsize=30)
            # axj.set_yticks(accuracy_ticks, minor=False)
            # g.tick_params(axis='y', which='major', labelsize=20)
            # plt.ylim(25, 60)
            # ax2 = g.twinx()
            # # y1_tick_positions = ax.get_ticks()
            # epsilon_ticks = original_performance - np.linspace(25, 55, len(ticks))
            # ax2.set_yticks(epsilon_ticks,minor=False)
            # ax2.set_ylabel(r"$\epsilon$", fontsize=20)
            # ax2.spines['right'].set_color('red')
            # ax2.tick_params(axis="y", colors="red",labelsize =15
            #                 )
            # ax2.yaxis.label.set_color('red')
            # ax2.invert_yaxis()

            handles, labels = axj.get_legend_handles_labels()
            l = axj.legend(handles[:5], labels[:5], fontsize=15)
            axj.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            axj.set_xlabel("")
            axj.tick_params(axis="y", labelsize=20)
            axj.set_ylabel("Accuracy", fontsize=20)
            # plt.ylim(25,55)
            l = axj.get_ylim()
            plt.ylim(l[0], l[1])
            ax2 = axj.twinx()
            ax2.set_ylim(l[0], l[1])
            # l2 = ax2.
            # f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
            # ticks = f(axj.get_yticks())
            unnormalied_ticks = axj.get_yticks()
            # ax2.set_yticks(axj.get_yticks(),labeget_ylim()
            #             # ax2.set_ylim(*l)ls=[25,30,35,40,45,50,55], minor=False)
            #
            # ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            # axj.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            y2_ticks = ax2.get_yticks()
            epsilon_ticks = original_performance - np.linspace(unnormalied_ticks[0], unnormalied_ticks[-1],
                                                               len(y2_ticks))
            new_ticks = np.linspace(l[0], l[1],
                                    len(y2_ticks))
            if l[1] - l[0] <= 2:
                formatter_q = lambda n: "{:10.2f}".format(n).replace(" ", "")
                formatter_f = lambda n: "{:10.2f}".format(n).replace(" ", "")
            else:
                formatter_q = lambda n: "{:10.0f}".format(n).replace(" ", "")
                formatter_f = lambda n: "{:10.1f}".format(n).replace(" ", "")
            new_ticks = list(map(formatter_q, unnormalied_ticks))
            axj.set_yticks(ticks=unnormalied_ticks, labels=new_ticks, minor=False)
            epsilon_ticks = list(map(formatter_f, epsilon_ticks))
            epsilon_ticks.reverse()
            ax2.set_yticks(ticks=unnormalied_ticks, labels=epsilon_ticks, minor=False)
            # ax2.set_yticks([25,30,35,40,45,50,55], minor=False)
            ax2.set_ylabel(r"$\epsilon$", fontsize=20)
            ax2.spines['right'].set_color('red')
            ax2.tick_params(axis="y", colors="red", labelsize=20
                            )
            ax2.yaxis.label.set_color('red')
            ax2.invert_yaxis()
            # ax2.tick_params(
            #     axis='x',  # changes apply to the x-axis
            #     which='both',  # both major and minor ticks are affected
            #     bottom=False,  # ticks along the bottom edge are off
            #     top=False,  # ticks along the top edge are off
            #     labelbottom=False)

            # plt.savefig("data/epsilon_allN_all_pr_{}_sigma={}.png".format(current_pr,current_sigma), bbox_inches="tight")
            # plt.savefig("data/epsilon_allN_all_pr_{}_sigma={}.pdf".format(current_pr,current_sigma), bbox_inches="tight")
        plt.tight_layout()
        pr_string = "_".join(list(map(str, specific_pruning_rates)))
        # plt.show()
        plt.savefig("data/epsilon_allN_all_pr_{}_sigma={}.png".format(pr_string, current_sigma), bbox_inches="tight")
        plt.savefig("data/epsilon_allN_all_pr_{}_sigma={}.pdf".format(pr_string, current_sigma), bbox_inches="tight")


def plot_val_accuracy_wandb(filepath, save_path, x_variable, y_variable, xlabel, ylabel):
    df = pd.read_csv(filepath_or_buffer=filepath, sep=",", header=0)
    df.plot(x=x_variable, y=y_variable, legend=False)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.savefig(save_path)


def plot_test_accuracy_wandb(filepaths,legends, save_path, x_variable, y_variable, y_min, y_max, xlabel, ylabel,
                             title=""):
    figure , ax = plt.subplots()
    for filepath in filepaths:
        df = pd.read_csv(filepath_or_buffer=filepath, sep=",", header=0)
        plt.plot(df[x_variable].ewm().mean(), df[y_variable].ewm())
        plt.fill_between(df[x_variable], df[y_min].ewm().mean(), df[y_max].man(), alpha=0.2)

    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    ax.legend(legends)
    plt.title(title, fontsize=20)
    # plt.savefig(save_path)
    plt.show()


def statistics_of_epsilon_for_stochastic_pruning(filepath: str, cfg: omegaconf.DictConfig):
    use_cuda = torch.cuda.is_available()

    net = get_model(cfg)

    _, _, testloader = get_cifar_datasets(cfg)

    original_performance = test(net, use_cuda, testloader)

    df = pd.read_csv(filepath_or_buffer=filepath, sep=",", header=0)
    num_col = len(df["Pruning Rate"].unique())
    num_row = len(df["sigma"].unique())
    # num_col = 3
    # num_row = 3
    # For each num_row is going to be one plot with num_col subplots for each pruning rate
    for i in range(num_row):
        current_sigma = df["sigma"].unique()[i]
        fig, axs = plt.subplots(1, num_col, figsize=(17, 10), layout="constrained")
        # plt.suptitle(r"$\sigma={}$".format(current_sigma), fontsize=20)
        for j, axj in enumerate(axs.flat):
            current_pr = df["Pruning Rate"].unique()[j]
            # axj = fig.add_subplot(1, num_col, j+1,)
            current_df = df[df["sigma"] == current_sigma]
            current_df = current_df[current_df["Pruning Rate"] == current_pr]

            pruned_original = copy.deepcopy(net)
            prune_with_rate(pruned_original, float(current_pr))
            remove_reparametrization(pruned_original)
            pruned_original_performance = test(pruned_original, use_cuda, testloader, verbose=0)
            delta_pruned_original_performance = original_performance - pruned_original_performance
            g = sns.boxplot(x='Type',
                            y='Epsilon',
                            hue="Type",
                            data=current_df,
                            ax=axj
                            )
            g.axhline(delta_pruned_original_performance, c="purple", label="Deterministic Pruning")
            g = sns.stripplot(x='Type',
                              y='Epsilon',
                              hue='Type',
                              jitter=True,
                              # dodge=True,
                              marker='o',
                              edgecolor="gray",
                              linewidth=1,
                              # palette="set2",
                              alpha=0.5,
                              data=current_df,
                              ax=axj)
            axj.set_title("Pruning Rate = {}".format(current_pr), fontsize=15)
            axj.set_ylabel("Epsilon".format(current_pr), fontsize=15)
            handles, labels = g.get_legend_handles_labels()
            l = axj.legend(handles[:4], labels[:4])
            axj.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            axj.set_xlabel("")

            # plt.savefig("data/epsilon_allN_all_pr_{}_sigma={}.png".format(current_pr,current_sigma), bbox_inches="tight")
            # plt.savefig("data/epsilon_allN_all_pr_{}_sigma={}.pdf".format(current_pr,current_sigma), bbox_inches="tight")
        plt.savefig("data/epsilon_allN_all_pr_sigma={}.png".format(current_sigma), bbox_inches="tight")
        plt.savefig("data/epsilon_allN_all_pr_sigma={}.pdf".format(current_sigma), bbox_inches="tight")

    # subfigs = fig.subfigures(num_row, 1)
    # for i, subfig in enumerate(subfigs):
    #     axes = subfig.subplots(1, num_col, sharey=True)
    #     string_formatted = ""
    #     current_sigma = df["sigma"].unique()[i]
    #     subfig.suptitle(r"$\sigma={}$".format(df["sigma"].unique()[i]))
    #     # subfig.suptitle(r"$\sigma={}$".format(0.00214))
    #     for j, ax in enumerate(axes):
    #         current_pr = df["Pruning Rate"].unique()[j]
    #         ax.set_title(f'Pruning Rate={df["Pruning Rate"].unique()[j]}', fontsize=12)
    #         # ax.set_title(f'Pruning Rate=0.8', fontsize=12)
    #         current_df = df[df["sigma"] == current_sigma]
    #         current_df = current_df[current_df["Pruning Rate"] == current_pr]
    #         sns.boxplot(ax=ax, data=current_df, x='Population', y='Epsilon', hue="Type")
    #
    #         # sns.catplot(ax=ax,x='Population',
    #         #             y='Epsilon',
    #         #             hue="Type",
    #         #             kind="box",
    #         #             legend=False,
    #         #             # height=6,
    #         #             # aspect=1.3,
    #         #             data=current_df)  # .set(title="Title")
    #         # # make grouped stripplot
    #         # sns.stripplot(ax=ax,x='Population',
    #         #               y='Epsilon',
    #         #               hue='Type',
    #         #               jitter=True,
    #         #               dodge=True,
    #         #               marker='o',
    #         #               # palette="Set2",
    #         #               alpha=0.5,
    #         #               data=current_df)
    #
    # plt.savefig("test.png")
    # ########## For N = 1000 ###########
    # all1000_performances = np.load("data/population_data/performances_gaussian_transfer_t_20-34-.npy")
    # labels1000 = np.load("data/population_data/labels_gaussian_transfer_t_20-34.npy")
    # deltas_det_to_sto1000 = np.load("data/population_data/deltas_gaussian_det_mask_to_sto_weights_deltas "
    #                                "_N_1000_t_20-34.npy")
    #
    # N = len(deltas_det_to_sto1000)
    # stochastic_deltas1000 = []
    # for i, label in enumerate(labels1000):
    #
    #     if label == "stochastic pruned":
    #         stochastic_deltas1000.append(original_performance - all1000_performances[i])
    #         if len(stochastic_deltas1000) == N:
    #             break


def test_figure():
    exercise = sns.load_dataset("exercise")
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    g = sns.boxplot(x="time", y="pulse", hue="kind", data=exercise, ax=ax1)  # pass ax1
    g = sns.stripplot(x="time", y="pulse", hue="kind",
                      jitter=True,
                      dodge=True,
                      marker='o',
                      edgecolor="gray",
                      linewidth=1,
                      # palette="set2",
                      alpha=0.5,
                      data=exercise, ax=ax1)  # pass ax1

    ax2 = fig.add_subplot(122)
    g = sns.boxplot(x="time", y="pulse", hue="kind", data=exercise, ax=ax2)  # pass ax2
    g = sns.stripplot(x="time",
                      y="pulse", hue="kind",
                      jitter=True,
                      dodge=True,
                      marker='o',
                      edgecolor="gray",
                      linewidth=1,
                      # palette="set2",
                      alpha=0.5,
                      data=exercise,
                      ax=ax2)  # pass ax1

    plt.close(2)
    plt.close(3)
    plt.tight_layout()
    plt.savefig("test.png")


def population_sweeps_transfer_mask_rank_experiments(identifier=""):
    cfg = omegaconf.DictConfig({
        "population": 3,
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
    Ns = [10, 50, 100]
    sigmas = np.linspace(start=0.001, stop=0.005, num=3)
    # sigmas = [0.0021419609859022197]
    pruning_rates = [0.5, 0.8, 0.9]
    df = pd.DataFrame(columns=["Epsilon", "Population", "Type", "Pruning Rate", "sigma"])
    result = time.localtime(time.time())
    file_path = f"data/epsilon_experiments_{identifier}_t_{result.tm_hour}-{result.tm_min}"
    for pop in Ns:
        for sig in sigmas:
            for pr in pruning_rates:
                cfg.population = pop
                cfg.sigma = float(sig)
                cfg.amount = pr
                df_result = transfer_mask_rank_experiments_no_plot(cfg)
                df = df.append(df_result)
                df_result.to_csv(file_path + f"pop_{pop}_sig_{sig}_pr_{pr}.csv", sep=",", index=False)
    df.to_csv(file_path + "_full.csv", sep=",", index=False)
    return file_path


############################# Stochastic pruning with sigma optimization ###########################################
def get_sigma_sample_per_layer_optuna(trial: optuna.Trial, lower_limit: int, upper_limit_per_layer: dict,
                                      exclude_layer_list: list = []):
    # trial = study.ask()  # `trial` is a `Trial` and not a `FrozenTrial`.
    new_dict = {}
    for key, value in upper_limit_per_layer.items():
        if key in exclude_layer_list:
            continue
        else:
            new_dict[key] = trial.suggest_float("{}_sigma".format(key), lower_limit, value, log=True)

    return trial, new_dict


def get_sigma_pr_sample_per_layer_optuna(trial: optuna.Trial, lower_limit: int, upper_limit_per_layer: dict,
                                         number_weights_per_layer: dict, total_number_weights: int):
    # trial = study.ask()  # `trial` is a `Trial` and not a `FrozenTrial`.
    sigma_dict = {}
    pr_dict = {}
    sum_of_pruning_weights_by_layer = 0
    for key, value in upper_limit_per_layer.items():
        sigma_dict[key] = trial.suggest_float("{}_sigma".format(key), lower_limit, value, log=True)
        pr_dict[key] = trial.suggest_float("{}_pr".format(key), 0.05, 0.99)
        sum_of_pruning_weights_by_layer += pr_dict[key] * number_weights_per_layer[key]
    # Store the constraints as user attributes so that they can be restored after optimization.
    c0 = total_number_weights - sum_of_pruning_weights_by_layer
    trial.set_user_attr("constraint", (c0,))
    return trial, sigma_dict, pr_dict


def get_pr_sample_per_layer_optuna(trial: optuna.Trial, layer_names: dict,
                                   number_weights_per_layer: dict, lower_bounds: dict, total_number_weights_pruned: int,
                                   exclude_layers=[]):
    # trial = study.ask()  # `trial` is a `Trial` and not a `FrozenTrial`.
    pr_dict = {}
    sum_of_pruning_weights_by_layer = 0
    for key in layer_names:
        if key in exclude_layers:
            sum_of_pruning_weights_by_layer += number_weights_per_layer[key]
            continue
        pr_dict[key] = trial.suggest_float("{}_pr".format(key), lower_bounds[key], 0.99)
        sum_of_pruning_weights_by_layer += pr_dict[key] * number_weights_per_layer[key]
    # Store the constraints as user attributes so that they can be restored after optimization.
    c0 = total_number_weights_pruned - sum_of_pruning_weights_by_layer
    trial.set_user_attr("constraint", (c0,))
    return trial, pr_dict, c0


def stochastic_pruning_with_sigma_and_pr_optimization(cfg: omegaconf.DictConfig):
    print("Config: \n{}".format(omegaconf.OmegaConf.to_yaml(cfg)))
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    N = cfg.population
    first_sparsity = 0.5
    original_model = get_model(cfg)
    if cfg.use_wandb:
        os.environ["WANDB_START_METHOD"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"pruning_rate_sigma_optimization",
            reinit=True,
            save_code=True,
        )

    _, pr_per_layer, baseline_non_zero, total_param = erdos_renyi_per_layer_pruning_rate(model=original_model, cfg=cfg)
    deterministic_pruning = copy.deepcopy(original_model)
    prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers)
    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)

    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    count = lambda w: w.nelement()
    number_per_layer = list(map(count, weights))
    number_per_layer = dict(zip(names, number_per_layer))

    ################# Optuna study###########################
    def constraints(trial):
        return trial.user_attrs["constraint"]

    sampler = optuna.integration.BoTorchSampler(
        constraints_func=constraints,
        n_startup_trials=10,
    )
    study = optuna.create_study(study_name="stochastic pruning with pr and sigma optimisation", sampler=sampler,
                                direction="maximize")
    total_surviving_weights = count_parameters(original_model) * cfg.amount

    quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
                                     names=["layer", "q25", "q50", "q75"])
    sigma_upper_bound_per_layer = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    lower_bound = 1e-10
    best_model_found = None
    best_accuracy_found = 0
    best_sigmas_found = None
    best_pruning_rates_found = None
    current_best_model = original_model

    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        current_gen_sigmas = []
        current_gen_prs = []
        for individual_index in range(N):
            individual_trial = study.ask()
            individual = copy.deepcopy(current_best_model)
            ############### Here I ask for pr and for sigma ###################################
            individual_trial, sigmas_for_individual, pr_for_individual = get_sigma_pr_sample_per_layer_optuna(
                individual_trial,
                lower_bound,
                sigma_upper_bound_per_layer,
                number_weights_per_layer=number_per_layer,
                total_number_weights=total_surviving_weights)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_individual)
            ############ Prune with optuna prs #######################################
            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner="manual", pr_per_layer=pr_per_layer)
            ##########################################################################################
            noisy_sample_performance = test(noisy_sample, use_cuda, valloader, verbose=0)
            study.tell(individual_trial, noisy_sample_performance)
            print("Generation {} Individual {}:{}".format(gen, individual_index, noisy_sample_performance))
            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
            current_gen_sigmas.append(sigmas_for_individual)
            current_gen_prs.append(pr_for_individual)

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            best_sigmas_found = current_gen_sigmas[best_index]
            best_pruning_rates_found = current_gen_prs[best_index]
            if cfg.use_wandb:
                wandb.log({"val_set_accuracy": best_accuracy_found, "generation": gen,
                           "sparsity": sparsity(best_model_found)})

            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found

            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))
    print("\n Best trial:")
    trial = study.best_trial

    print("  Performance on validation set for set of sigmas: {}".format(trial.value))

    print("Set of sigmas and pruning rates: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    accuracy_string = "{:10.2f}%".format(performance_best_model_found)
    result = time.localtime(time.time())
    ######Saving the model the pruning
    with open(cfg.save_model_path + "stochastic_pruning_pr_optim_test_accuracy={}_time_{}-{}.pth".format(
            accuracy_string, result.tm_hour, result.tm_min),
              "wb") as f:
        pickle.dump(best_model_found, f)
    with open(cfg.save_data_path + "sigmas_per_layers_pr_optim_time_{}-{}.pth".format(result.tm_hour,
                                                                                      result.tm_min),
              "wb") as f:
        pickle.dump(best_sigmas_found, f)
    with open(cfg.save_data_path + "pruning_rate_per_layers_pr_optim_time_{}-{}.pth".format(result.tm_hour,
                                                                                            result.tm_min),
              "wb") as f:
        pickle.dump(best_pruning_rates_found, f)

    if cfg.use_wandb:
        wandb.join()


def stochastic_pruning_with_sigma_optimization_with_erk_layer_wise_prunig_rates(cfg: omegaconf.DictConfig):
    print("Config: \n{}".format(omegaconf.OmegaConf.to_yaml(cfg)))
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"erk_pr_sigma_optimization",
            reinit=True,
            save_code=True,
        )
    N = cfg.population
    first_sparsity = 0.5
    original_model = get_model(cfg)

    _, pr_per_layer, _, total_param = erdos_renyi_per_layer_pruning_rate(model=original_model, cfg=cfg)
    deterministic_pruning = copy.deepcopy(original_model)

    prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers)
    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    noise = [cfg.sigma] * len(names)
    noise_per_layer = dict(zip(names, noise))
    ############## Craete the study###############################
    study = optuna.create_study(study_name="stochastic pruning "
                                           "with sigma optimization with erk layer wise prunig rates",
                                direction="maximize")

    quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
                                     names=["layer", "q25", "q50", "q75"])
    sigma_upper_bound_per_layer = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    lower_bound = 1e-10
    best_model_found = None
    best_accuracy_found = 0
    best_sigma_found = None
    current_best_model = original_model
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        current_gen_sigmas = []
        for individual_index in range(N):
            individual_trial = study.ask()
            individual = copy.deepcopy(current_best_model)

            individual_trial, sigmas_for_individual = get_sigma_sample_per_layer_optuna(individual_trial, lower_bound,
                                                                                        sigma_upper_bound_per_layer,
                                                                                        exclude_layer_list=cfg.exclude_layers)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_individual)
            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner="erk")
            noisy_sample_performance = test(noisy_sample, use_cuda, valloader, verbose=0)
            study.tell(individual_trial, noisy_sample_performance)
            print("Generation {} Individual {} sparsity {}:{}".format(gen, individual_index, sparsity(noisy_sample),
                                                                      noisy_sample_performance))
            # print("Sigmas used:")
            # print(omegaconf.OmegaConf.to_yaml(sigmas_for_individual))

            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
            current_gen_sigmas.append(sigmas_for_individual)

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            best_sigma_found = current_gen_sigmas[best_index]
            if cfg.use_wandb:
                wandb.log({"val_set_accuracy": best_accuracy_found, "generation": gen,
                           "sparsity": sparsity(best_model_found)})

            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found

            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp
        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))
    print("\n Best trial:")
    trial = study.best_trial

    print("  Performance on validation set for set of sigmas: {}".format(trial.value))

    print("Set of sigmas: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    accuracy_string = "{:10.2f}%".format(performance_best_model_found)
    result = time.localtime(time.time())
    with open(cfg.save_model_path + "stochastic_pruning_erdos_renyi_dist_test_accuracy={}_time_{}-{}.pth".format(
            accuracy_string, result.tm_hour, result.tm_min),
              "wb") as f:
        pickle.dump(best_model_found, f)
    with open(cfg.save_data_path + "sigmas_per_layers_erdos_renyi_dist_time_{}-{}.pth".format(result.tm_hour,
                                                                                              result.tm_min),
              "wb") as f:
        pickle.dump(best_sigma_found, f)
    if cfg.use_wandb:
        wandb.join()


def enqueue_sigmas(study: optuna.Study, initial_sigmas_per_layer: dict):
    first_trial_dict = {}
    for key, value in initial_sigmas_per_layer.items():
        first_trial_dict["{}_sigma".format(key)] = value
    study.enqueue_trial(first_trial_dict, user_attrs={"memo": "good global value fond by grid "
                                                              "search"})


def enqueue_pr(study: optuna.Study, initial_pr_per_layer: dict):
    first_trial_dict = {}
    for key, value in initial_pr_per_layer.items():
        first_trial_dict[key] = float(value)
    print(omegaconf.OmegaConf.to_yaml(first_trial_dict))
    study.enqueue_trial(first_trial_dict, user_attrs={"memo": "good pr found from literature"})


############################# Ablation experiments #####################################################################


def dynamic_sigma_per_layer_one_shot_pruning(cfg: omegaconf.DictConfig):
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"

        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"one-shot_{cfg.pruner}_pr_{target_sparsity}_sampler_{cfg.sampler}_sigma_optimization",
            reinit=True,
            save_code=True,
        )
    N = cfg.population
    original_model = get_model(cfg)

    _, pr_per_layer, _, total_param = erdos_renyi_per_layer_pruning_rate(model=original_model, cfg=cfg)
    deterministic_pruning = copy.deepcopy(original_model)

    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))
    print("Deterministic pruning sparsity:  {}".format(sparsity(deterministic_pruning)))

    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    noise_upper_bound = [0.01] * len(names)
    # This initial value is good for pruning rates >= 0.8. For pruning rates like 0.5 is not that good
    noise_initial_value = [0.005] * len(names)
    noise_upper_bound_dict = dict(zip(names, noise_upper_bound))
    noise_initail_value_dict = dict(zip(names, noise_initial_value))
    ############## Craete the study###############################
    sampler = None
    sampler = get_sampler(cfg)
    study = optuna.create_study(study_name="One shot stochastic pruning "
                                           "with sigma optimization per layer",
                                direction="maximize", sampler=sampler)
    ######### Here I set the intial values I want the trial to have which is a global value
    enqueue_sigmas(study, noise_initail_value_dict)
    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])

    # noise_initail_value_dict = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    lower_bound = 1e-10
    best_model_found = None
    best_accuracy_found = 0
    best_sigma_found = None
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        current_gen_sigmas = []
        best_gen_accuracy = 0
        best_gen_model = None
        best_gen_sigma = None

        for individual_index in range(N):
            individual_trial = study.ask()
            individual = copy.deepcopy(original_model)

            individual_trial, sigmas_for_individual = get_sigma_sample_per_layer_optuna(individual_trial, lower_bound,
                                                                                        noise_upper_bound_dict,
                                                                                        exclude_layer_list=cfg.exclude_layers)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_individual)
            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)
            noisy_sample_performance = test(noisy_sample, use_cuda, valloader, verbose=0)
            study.tell(individual_trial, noisy_sample_performance)
            print("Generation {} Individual {} sparsity {}:{}".format(gen, individual_index, sparsity(noisy_sample),
                                                                      noisy_sample_performance))

            # print(omegaconf.OmegaConf.to_yaml(sigmas_for_individual))
            if noisy_sample_performance > best_gen_accuracy:
                best_gen_accuracy = noisy_sample_performance
                best_gen_model = noisy_sample
                best_gen_sigma = sigmas_for_individual

            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
            current_gen_sigmas.append(sigmas_for_individual)

        # best_index = np.argmax(current_gen_accuracies)
        # gen_best_accuracy = current_gen_accuracies[best_index]

        if cfg.use_wandb:
            log_dict = {"val_set_accuracy": best_gen_accuracy, "generation": gen,
                        "sparsity": sparsity(best_gen_model),
                        "Deterministic performance": deterministic_pruning_performance}
            log_dict.update(best_gen_sigma)
            wandb.log(log_dict)
        if best_gen_accuracy > best_accuracy_found:
            best_accuracy_found = best_gen_accuracy
            best_model_found = best_gen_model
            best_sigma_found = best_gen_sigma

            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": performance_best_model_found, "generation": cfg.generations - 1,
                   "sparsity": sparsity(best_model_found),
                   "Deterministic performance": deterministic_pruning_performance})
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {}"
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))
    print("\n Best trial:")
    trial = study.best_trial

    print("  Performance on validation set for set of sigmas: {}".format(trial.value))

    print("Set of sigmas: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    accuracy_string = "{:10.2f}".format(performance_best_model_found)
    result = time.localtime(time.time())
    model_name = cfg.save_model_path + "one_shot_dynamic_sigmas_stochastic_pruning_{}_dist_test_accuracy={}_time_{}-{}.pth".format(
        cfg.pruner, accuracy_string, result.tm_hour, result.tm_min)
    model_name = model_name.replace(" ", "")
    with open(model_name, "wb") as f:
        pickle.dump(best_model_found, f)
    with open(cfg.save_data_path + "one_shot_dynamic_sigmas_per_layers_{}_dist_pr_{}_sampler_{}.pth".format(cfg.pruner,
                                                                                                            cfg.amount,
                                                                                                            cfg.sampler),
              "wb") as f:
        pickle.dump(best_sigma_found, f)
    if cfg.use_wandb:
        wandb.save(model_name)
        wandb.join()


def static_sigma_per_layer_manually_iterative_process(cfg: omegaconf.DictConfig):
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"iterative_{cfg.pruner}_pr_{cfg.amount}_sigma_manual_10_percentile",
            reinit=True,
            save_code=True,
        )
    N = cfg.population
    original_model = get_model(cfg)
    deterministic_pruning = copy.deepcopy(original_model)
    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    sigmas_for_experiment = get_percentile_per_layer(original_model, 0.1)

    # This initial value is good for pruning rates >= 0.8. For pruning rates like 0.5 is not that good

    ############## Craete the study###############################

    ######### Here I set the intial values I want the trial to have which is a global value
    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])

    # noise_initail_value_dict = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    best_model_found = None
    best_accuracy_found = 0
    current_best_model = original_model
    total_sparse_flops = 0
    first_time = 1
    iter_val_loader = cycle(iter(valloader))
    data, y = next(iter_val_loader)
    _, unit_sparse_flops = flops(deterministic_pruning, data)
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        batch_for_gen = [next(iter_val_loader)]
        for individual_index in range(N):
            individual = copy.deepcopy(current_best_model)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_experiment)
            if cfg.pruner == "global":
                prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)
            remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)
            noisy_sample_performance, individual_sparse_flops = test(noisy_sample, use_cuda, batch_for_gen, verbose=0,
                                                                     count_flops=True, batch_flops=unit_sparse_flops,
                                                                     one_batch=cfg.one_batch)
            total_sparse_flops += individual_sparse_flops
            print("Generation {} Individual {} sparsity {:0.3f} FLOPS {} Accuracy {}".format(gen, individual_index,
                                                                                             sparsity(noisy_sample),
                                                                                             sparse_flops,
                                                                                             noisy_sample_performance))

            # print(omegaconf.OmegaConf.to_yaml(sigmas_for_individual))

            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if cfg.use_wandb:
            test_accuracy = test(current_gen_models[best_index], use_cuda, [get_random_batch(testloader)], verbose=0,
                                 one_batch=cfg.one_batch)
            log_dict = {"val_set_accuracy": gen_best_accuracy, "generation": gen,
                        "sparsity": sparsity(current_gen_models[best_index]),
                        "Deterministic performance": deterministic_pruning_performance,
                        "sparse_flops": total_sparse_flops,
                        "test_set_accuracy": test_accuracy}
            wandb.log(log_dict)
        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found
            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": performance_best_model_found, "generation": cfg.generations - 1,
                   "sparsity": sparsity(best_model_found),
                   "Deterministic performance": deterministic_pruning_performance})
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))

    accuracy_string = "{:10.2f}%".format(performance_best_model_found).replace(" ", "")
    result = time.localtime(time.time())
    model_file_name = cfg.save_model_path + "one_shot_manual_sigma_stochastic_pruning_{}_dist_test_accuracy={}_time_{}-{}.pth".format(
        cfg.pruner, accuracy_string, result.tm_hour, result.tm_min)
    model_file_name = model_file_name.replace(" ", "")
    with open(model_file_name, "wb") as f:
        pickle.dump(best_model_found, f)

    if cfg.use_wandb:
        wandb.save(model_file_name)
        wandb.join()


def static_sigma_per_layer_optimized_iterative_process(cfg: omegaconf.DictConfig):
    sigmas_for_experiment = None
    try:
        print(cfg.save_data_path + "one_shot_dynamic_sigmas_per_layers_{}_dist_pr_{}_sampler_{}.pth".format(cfg.pruner,
                                                                                                            cfg.amount,
                                                                                                            cfg.sampler))
        with open(cfg.save_data_path + "one_shot_dynamic_sigmas_per_layers_{}_dist_pr_{}_sampler_{}.pth".format(
                cfg.pruner,
                cfg.amount, cfg.sampler),
                  "rb") as f:

            sigmas_for_experiment = pickle.load(f)
    except:
        raise Exception("The respective sigmas file for pruning rate {} and pr per layer {} and sampler {} hasn't been "
                        "generated, "
                        "You need to "
                        "run first \n"
                        "dynamic_sigma_per_layer_one_shot_pruning function with the desired pruning rate and pr per layer".format(
            cfg.amount, cfg.pruner, cfg.sampler))
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    print("Cuda: {}".format(use_cuda))
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"iterative_{cfg.pruner}_pr_{cfg.amount}_sigma_optimized",
            reinit=True,
            save_code=True,
        )
    N = cfg.population
    original_model = get_model(cfg)

    _, pr_per_layer, _, total_param = erdos_renyi_per_layer_pruning_rate(model=original_model, cfg=cfg)
    deterministic_pruning = copy.deepcopy(original_model)

    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="layer-wise",
                        pruner=cfg.pruner)
    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    # This initial value is good for pruning rates >= 0.8. For pruning rates like 0.5 is not that good

    ############## Craete the study###############################

    ######### Here I set the intial values I want the trial to have which is a global value
    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])

    # noise_initail_value_dict = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    best_model_found = None
    best_accuracy_found = 0
    current_best_model = original_model
    iter_val_loader = cycle(iter(valloader))
    data, y = next(iter_val_loader)
    _, unit_sparse_flops = flops(deterministic_pruning, data)
    total_sparse_flops = 0
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        batch_for_gen = [next(iter_val_loader)]
        for individual_index in range(N):
            individual = copy.deepcopy(current_best_model)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_experiment)
            ############################# Pruning ##################################################
            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)
            remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)
            noisy_sample_performance, individual_sparse_flops = test(noisy_sample, use_cuda, batch_for_gen, verbose=0,
                                                                     count_flops=True, batch_flops=unit_sparse_flops,
                                                                     one_batch=cfg.one_batch)
            total_sparse_flops += individual_sparse_flops
            print("Generation {} Individual {} sparsity {:0.3f} FLOPS {} Accuracy {}".format(gen, individual_index,
                                                                                             sparsity(noisy_sample),
                                                                                             sparse_flops,
                                                                                             noisy_sample_performance))

            # print(omegaconf.OmegaConf.to_yaml(sigmas_for_individual))

            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if cfg.use_wandb:
            test_accuracy = test(current_gen_models[best_index], use_cuda, [get_random_batch(testloader)], verbose=0,
                                 one_batch=cfg.one_batch)
            log_dict = {"val_set_accuracy": gen_best_accuracy, "generation": gen,
                        "sparsity": sparsity(current_gen_models[best_index]),
                        "Deterministic performance": deterministic_pruning_performance,
                        "sparse_flops": total_sparse_flops,
                        "test_set_accuracy": test_accuracy}
            wandb.log(log_dict)

        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found
            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": performance_best_model_found, "generation": cfg.generations - 1,
                   "sparsity": sparsity(best_model_found),
                   "Deterministic performance": deterministic_pruning_performance})
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))

    accuracy_string = "{:10.2f}".format(performance_best_model_found)
    result = time.localtime(time.time())
    model_file_name = cfg.save_model_path + "iterative_static_sigma_optimized_stochastic_pruning_{}_dist_test_accuracy={}_time_{}-{}.pth".format(
        cfg.pruner,
        accuracy_string, result.tm_hour, result.tm_min).replace(" ", "")
    with open(model_file_name,
              "wb") as f:
        pickle.dump(best_model_found, f)
    if cfg.use_wandb:
        wandb.save(model_file_name)
        wandb.join()


def static_global_sigma_iterative_process(cfg: omegaconf.DictConfig):
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    one_batch_string = "_one_batch_per_generation" if cfg.one_batch else "_whole_valset_per_generation "
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"iterative_{cfg.pruner}_pr_{cfg.amount}_global_static_sigma{one_batch_string}",
            reinit=True,
            save_code=True,
        )
    N = cfg.population
    original_model = get_model(cfg)

    _, pr_per_layer, _, total_param = erdos_renyi_per_layer_pruning_rate(model=original_model, cfg=cfg)
    deterministic_pruning = copy.deepcopy(original_model)
    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)
    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    # This initial value is good for pruning rates >= 0.8. For pruning rates like 0.5 is not that good

    ############## Craete the study###############################

    ######### Here I set the intial values I want the trial to have which is a global value
    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])

    # noise_initail_value_dict = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    data_loader_iterator = cycle(iter(valloader))
    best_model_found = None
    best_accuracy_found = 0
    current_best_model = original_model
    data, y = next(data_loader_iterator)
    _, unit_sparse_flops = flops(deterministic_pruning, data)
    iter_val_loader = cycle(iter(valloader))
    total_sparse_flops = 0
    first_generation = True
    evaluation_set = valloader
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []

        if cfg.one_batch:
            evaluation_set = [next(iter_val_loader)]
        ################################################################################################################
        for individual_index in range(N):
            individual = copy.deepcopy(current_best_model)

            noisy_sample = get_noisy_sample(individual, cfg)

            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)

            noisy_sample_performance, sparse_flops = test(noisy_sample, use_cuda, evaluation_set, verbose=0,
                                                          count_flops=True, batch_flops=unit_sparse_flops)
            total_sparse_flops += sparse_flops
            print("Generation {} Individual {} sparsity {}:{}".format(gen, individual_index, sparsity(noisy_sample),
                                                                      noisy_sample_performance))
            # print(omegaconf.OmegaConf.to_yaml(sigmas_for_individual))
            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
        ################################################################################################################
        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if cfg.use_wandb:
            test_accuracy = test(current_gen_models[best_index], use_cuda, [get_random_batch(testloader)], verbose=0,
                                 one_batch=cfg.one_batch)
            log_dict = {"val_set_accuracy": gen_best_accuracy, "generation": gen,
                        "sparsity": sparsity(current_gen_models[best_index]),
                        "Deterministic performance": deterministic_pruning_performance,
                        "sparse_flops": total_sparse_flops,
                        "test_set_accuracy": test_accuracy,
                        "sigma": cfg.sigma
                        }
            wandb.log(log_dict)

        if first_generation:
            cfg.sigma = cfg.sigma / 2
            first_generation = False
        if gen == int((1 / 3) * cfg.generations):
            cfg.sigma = cfg.sigma / 2
        if gen == int((2 / 3) * cfg.generations):
            cfg.sigma = cfg.sigma / 2

        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found
            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": performance_best_model_found, "generation": cfg.generations - 1,
                   "sparsity": sparsity(best_model_found),
                   "Deterministic performance": deterministic_pruning_performance})
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))

    accuracy_string = "{:10.2f}%".format(performance_best_model_found)
    result = time.localtime(time.time())
    model_file_name = cfg.save_model_path + "global_sigma_iterative_stochastic_pruning_pr_{}_{}_dist_test_accuracy={}_time_{}-{}.pth".format(
        cfg.amount, cfg.pruner,
        accuracy_string, result.tm_hour, result.tm_min)
    model_file_name = model_file_name.replace(" ", "")
    with open(model_file_name, "wb") as f:
        pickle.dump(best_model_found, f)
    if cfg.use_wandb:
        wandb.save(model_file_name)
        wandb.join()


def dynamic_sigma_iterative_process(cfg: omegaconf.DictConfig, print_exclude_layers):
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    N = cfg.population
    generations = cfg.generations
    original_model = get_model(cfg)
    deterministic_pruning = copy.deepcopy(original_model)
    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="layer-wise",
                        pruner=cfg.pruner)
    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"

        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"iterative_{cfg.pruner}_pr_{target_sparsity}_sampler_{cfg.sampler}_dynamic_sigma",
            reinit=True,
            save_code=True,
        )
    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    noise_upper_bound = [0.01] * len(names)
    # This initial value is good for pruning rates >= 0.8. For pruning rates like 0.5 is not that good
    noise_initial_value = [0.005] * len(names)
    noise_upper_bound_dict = dict(zip(names, noise_upper_bound))
    noise_initail_value_dict = dict(zip(names, noise_initial_value))
    ############## Craete the study###############################
    study = optuna.create_study(study_name="Iterative stochastic pruning "
                                           "with sigma optimization per layer",
                                direction="maximize")
    ######### Here I set the intial values I want the trial to have which is a global value
    enqueue_sigmas(study, noise_initail_value_dict)
    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])

    # noise_initail_value_dict = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    lower_bound = 1e-10
    best_model_found = None
    best_accuracy_found = 0
    best_sigma_found = None
    current_best_model = original_model
    for gen in range(generations):
        current_gen_models = []
        current_gen_accuracies = []
        current_gen_sigmas = []
        for individual_index in range(N):
            individual_trial = study.ask()
            individual = copy.deepcopy(current_best_model)
            individual_trial, sigmas_for_individual = get_sigma_sample_per_layer_optuna(individual_trial, lower_bound,
                                                                                        noise_upper_bound_dict,
                                                                                        exclude_layer_list=cfg.exclude_layers)
            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_individual)
            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)
            noisy_sample_performance = test(noisy_sample, use_cuda, valloader, verbose=0)

            study.tell(individual_trial, noisy_sample_performance)
            print("Generation {} Individual {}:{}".format(gen, individual_index, noisy_sample_performance))
            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
            current_gen_sigmas.append(sigmas_for_individual)

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if cfg.use_wandb:
            log_dict = {"val_set_accuracy": gen_best_accuracy, "generation": gen,
                        "sparsity": sparsity(current_gen_models[best_index]),
                        "Deterministic performance": deterministic_pruning_performance}
            log_dict.update(current_gen_sigmas[best_index])
            wandb.log(log_dict)

        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            best_sigma_found = current_gen_sigmas[best_index]

            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found

            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp
        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": performance_best_model_found, "generation": cfg.generations - 1,
                   "sparsity": sparsity(best_model_found),
                   "Deterministic performance": deterministic_pruning_performance})
    print("The performance of on test set the best model is {} with pruning rate of {}".format(
        performance_best_model_found, cfg.amount))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))
    ############################################################################
    print("\n Best trial:")
    trial = study.best_trial

    print("  Performance on validation set for set of sigmas: {}".format(trial.value))
    print("Set of sigmas: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    ######################### Saving the model and the sigmas ##########################################################
    accuracy_string = "{:10.2f}".format(performance_best_model_found)
    result = time.localtime(time.time())
    model_file_name = cfg.save_model_path + "iterative_dynamic_sigma_stochastic_pruning_{}_dist_test_accuracy={}_time_{}-{}.pth".format(
        cfg.pruner, accuracy_string, result.tm_hour, result.tm_min)
    model_file_name = model_file_name.replace(" ", "")
    with open(model_file_name, "wb") as f:
        pickle.dump(best_model_found, f)
    with open(cfg.save_data_path + "iterative_dynamic_sigmas_per_layers_{}_dist_pr_{}.pth".format(cfg.pruner,
                                                                                                  cfg.amount),
              "wb") as f:
        pickle.dump(best_sigma_found, f)
    if cfg.use_wandb:
        wandb.save(
            cfg.save_model_path + "iterative_dynamic_sigma_stochastic_pruning_{}_dist_test_accuracy={}_time_{}-{}.pth".format(
                cfg.pruner, accuracy_string, result.tm_hour, result.tm_min))
        wandb.join()

def compare_weights(weight_list_1:List[torch.TensorType],weight_list_2:List[torch.TensorType]):
    list_equals = []
    list_of_elem = []
    for i ,weight1 in enumerate(weight_list_1):
        weight2 = weight_list_2[i]
        list_equals.append((weight1==weight2).sum())
        list_of_elem.append(weight2.numel())
    return list_equals,list_of_elem


def run_fine_tune_experiment(cfg: omegaconf.DictConfig):
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    exclude_layers_string = "_exclude_layers_fine_tuned" if cfg.fine_tune_exclude_layers else ""
    non_zero_string = "_non_zero_weights_fine_tuned" if cfg.fine_tune_non_zero_weights else ""
    post_pruning_noise_string = "_post_training_noise" if bool(cfg.noise_after_pruning)*cfg.measure_gradient_flow else ""

    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"fine_tune_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}{post_pruning_noise_string}",
            reinit=True,
        )
    pruned_model = get_model(cfg)
    if cfg.pruner == "global":
        prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)
    # Add small noise just to get tiny variations of the deterministic case
    initial_performance = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=0)
    print("Original version performance: {}".format(initial_performance))
    if cfg.noise_after_pruning and cfg.measure_gradient_flow:
        remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)
        mask_dict = get_mask(pruned_model)
        names,weights = zip(*get_layer_dict(pruned_model))
        sigma_per_layer = dict(zip(names,[cfg.noise_after_pruning]*len(names)))
        p2_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
        print("p2_model version 1 performance: {}".format(initial_performance))
        apply_mask(p2_model,mask_dict)
        initial_performance = test(p2_model, use_cuda=use_cuda, testloader=testloader, verbose=0)
        print("p2_model version 2 performance: {} with sparsity {}".format(initial_performance,sparsity(p2_model)))
        pruned_model = p2_model

    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)
    # mask_dict = get_mask(pruned_model)
    # p2_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
    # print("p2_model version 1 performance: {}".format(initial_performance))
    # apply_mask(p2_model,mask_dict)
    # initial_performance = test(p2_model, use_cuda=use_cuda, testloader=testloader, verbose=0)
    # print("p2_model version 2 performance: {}".format(initial_performance))
    # return
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": initial_performance, "initial_accuracy": initial_performance})
    filepath_GF_measure =""
    if cfg.measure_gradient_flow:

        identifier = f"{time.time():14.2f}".replace(" ", "")
        if cfg.pruner == "lamp":
            filepath_GF_measure += "gradient_flow_data/deterministic_LAMP/{}/sigma{}/pr{}/{}/".format(cfg.architecture,cfg.sigma,cfg.amount,identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
                # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
            # else:
                # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
        if cfg.pruner == "global":
            filepath_GF_measure += "gradient_flow_data/deterministic_GLOBAL/{}/sigma{}/pr{}/{}/".format(cfg.architecture,cfg.sigma,cfg.amount,identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
            # else:
            #     filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"

    restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                       use_wandb=cfg.use_wandb, epochs=cfg.epochs, exclude_layers=cfg.exclude_layers,
                                       fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                       fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                       cfg=cfg,
                                       gradient_flow_file_prefix=filepath_GF_measure)

    if cfg.use_wandb:
        wandb.join()


def static_sigma_per_layer_manually_iterative_process_flops_counts(cfg: omegaconf.DictConfig, FLOP_limit: float = 1e15):
    FLOP_limit = cfg.flop_limit
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    one_batch_string = "_one_batch" if cfg.one_batch else ""
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"iterative_{cfg.pruner}_pr_{cfg.amount}_sigma_manual_10_percentile_flops_count{one_batch_string}"
                 f"{cfg.one_batch}",
            notes="This run use the global iterative first generation uses global sigma 0.005 and then uses the "
                  "10th percentile "
                  "of each layer",
            reinit=True,
            save_code=True,
        )
    N = cfg.population
    original_model = get_model(cfg)
    deterministic_pruning = copy.deepcopy(original_model)
    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    sigmas_for_experiment = get_percentile_per_layer(original_model, 0.1)

    # This initial value is good for pruning rates >= 0.8. For pruning rates like 0.5 is not that good

    ############## Craete the study###############################

    ######### Here I set the intial values I want the trial to have which is a global value
    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])

    # noise_initail_value_dict = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    best_model_found = None
    best_accuracy_found = 0
    current_best_model = original_model
    sparse_flops = 0
    first_time = 1
    iter_val_loader = cycle(iter(valloader))
    data, y = next(iter_val_loader)
    _, unit_sparse_flops = flops(deterministic_pruning, data)
    evaluation_set = valloader
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        if cfg.one_batch:
            evaluation_set = [next(iter_val_loader)]
        for individual_index in range(N):
            individual = copy.deepcopy(current_best_model)
            ###### If the pruning is vanila global then the first gen is with global sigma
            #### after that we do it with the 10th percentile iterative process
            if gen == 0 and cfg.pruner == "global":
                noisy_sample = get_noisy_sample(individual, cfg)
            else:
                noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_experiment)
            ############################## we prune #######################################
            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)

            #######################################################33
            remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)

            noisy_sample_performance, individual_sparse_flops = test(noisy_sample, use_cuda, evaluation_set, verbose=0,
                                                                     count_flops=True, batch_flops=unit_sparse_flops)
            sparse_flops += individual_sparse_flops
            print("Generation {} Individual {} sparsity {:0.3f} FLOPS {} Accuracy {}".format(gen, individual_index,
                                                                                             sparsity(noisy_sample),
                                                                                             sparse_flops,
                                                                                             noisy_sample_performance))
            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
            if FLOP_limit != 0:
                if sparse_flops > FLOP_limit:
                    break

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if cfg.use_wandb:
            test_accuracy = test(current_gen_models[best_index], use_cuda, [get_random_batch(testloader)], verbose=0)
            log_dict = {"val_set_accuracy": gen_best_accuracy, "generation": gen,
                        "sparsity": sparsity(current_gen_models[best_index]),
                        "Deterministic performance": deterministic_pruning_performance,
                        "sparse_flops": sparse_flops,
                        "test_set_accuracy": test_accuracy
                        }
            wandb.log(log_dict)
            if FLOP_limit != 0:
                if sparse_flops > FLOP_limit:
                    break
        if gen_best_accuracy > best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found
            temp = copy.deepcopy(best_model_found)
            # remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"test_set_accuracy": performance_best_model_found, "generation": cfg.generations - 1,
                   "sparsity": sparsity(best_model_found),
                   "Deterministic performance": deterministic_pruning_performance,
                   "sparse_flops": sparse_flops
                   })
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))
    #
    accuracy_string = "{:10.2f}".format(performance_best_model_found).replace(" ", "")
    result = time.localtime(time.time())
    model_file_name = cfg.save_model_path + "iterative_manual_sigma_{}_dist_test_accuracy={}_flops_{}_time_{}-{}.pth".format(
        cfg.pruner, accuracy_string, sparse_flops, result.tm_hour, result.tm_min)
    model_file_name = model_file_name.replace(" ", "")
    with open(model_file_name, "wb") as f:
        pickle.dump(best_model_found, f)

    if cfg.use_wandb:
        wandb.save(model_file_name)
        wandb.join()


def stochastic_pruning_with_static_sigma_and_pr_optimization(cfg: omegaconf.DictConfig):
    print("Config: \n{}".format(omegaconf.OmegaConf.to_yaml(cfg)))
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    N = cfg.population
    first_sparsity = 0.5
    original_model = get_model(cfg)
    if cfg.use_wandb:
        os.environ["WANDB_START_METHOD"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"iterative_dynamic_prpl_pr_{cfg.amount}_sigma_manual_10_percentile_flops_count",
            reinit=True,
            save_code=True,
        )

    _, pr_lowerbound_per_layer, baseline_non_zero, total_param = erdos_renyi_per_layer_pruning_rate(
        model=original_model, cfg=cfg)
    deterministic_pruning = copy.deepcopy(original_model)

    if cfg.pruner == "global":
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(deterministic_pruning, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    deterministic_pruning_performance = test(deterministic_pruning, use_cuda, testloader, verbose=0)
    ###############  Sigmas of 10th percentile  #################
    sigmas_for_experiment = get_percentile_per_layer(original_model, 0.1)

    print("Deterministic pruning performance: {}".format(deterministic_pruning_performance))

    ######## Begin the procedure    #################################

    names, weights = zip(*get_layer_dict(original_model))
    count = lambda w: w.nelement()
    number_per_layer = list(map(count, weights))
    number_per_layer = dict(zip(names, number_per_layer))

    ################# Optuna study###########################
    #### Good values of PR per layer dictated by LAMP strategy
    initial_pr_per_layer = prune_with_rate(copy.deepcopy(original_model), target_sparsity,
                                           exclude_layers=cfg.exclude_layers, type="layer-wise", pruner="lamp",
                                           return_pr_per_layer=True)
    initial_pr_per_layer = {f'{k}_pr': v for k, v in initial_pr_per_layer.items()}

    def constraints(trial):
        return trial.user_attrs["constraint"]

    sampler = optuna.samplers.NSGAIISampler(
        constraints_func=constraints,
    )
    study = optuna.create_study(study_name="stochastic pruning with pr optimisation", sampler=sampler,
                                # directions=["maximize","minimize"])
                                direction="maximize")

    ###### Put the first pr in initial #########
    enqueue_pr(study, initial_pr_per_layer)
    total_pruned_weights = count_parameters(original_model) * (cfg.amount - 0.02)

    # quantile_per_layer = pd.read_csv("data/quantiles_of_weights_magnitude_per_layer.csv", sep=",", header=1, skiprows=1,
    #                                  names=["layer", "q25", "q50", "q75"])
    # sigma_upper_bound_per_layer = quantile_per_layer.set_index('layer')["q25"].T.to_dict()
    # lower_bound = 1e-10
    best_model_found = None
    best_accuracy_found = 0
    best_sigmas_found = None
    best_pruning_rates_found = None
    current_best_model = original_model
    data_loader_iterator = cycle(iter(valloader))
    data, y = next(data_loader_iterator)
    _, unit_sparse_flops = flops(deterministic_pruning, data)
    total_sparse_flops = 0
    for gen in range(cfg.generations):
        current_gen_models = []
        current_gen_accuracies = []
        current_gen_prs = []
        sample_for_generation = [next(data_loader_iterator)]
        for individual_index in range(N):
            individual_trial = study.ask()
            individual = copy.deepcopy(current_best_model)
            ############### Here I ask for pr ###################################
            individual_trial, pr_for_individual, constrain_value = get_pr_sample_per_layer_optuna(
                individual_trial,
                names, lower_bounds=pr_lowerbound_per_layer,
                number_weights_per_layer=number_per_layer,
                total_number_weights_pruned=total_pruned_weights, exclude_layers=cfg.exclude_layers)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_experiment)

            ############ Prune with optuna prs #######################################

            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner="manual", pr_per_layer=pr_for_individual)
            ##########################################################################################
            noisy_sample_performance, sparse_flops = test(noisy_sample, use_cuda, sample_for_generation, verbose=0,
                                                          count_flops=True, batch_flops=unit_sparse_flops,
                                                          one_batch=cfg.one_batch)
            total_sparse_flops += sparse_flops
            # study.tell(individual_trial, [noisy_sample_performance, sparsity(noisy_sample)])
            study.tell(individual_trial, noisy_sample_performance)

            print("Generation {} Individual {} sparsity {}:{}".format(gen, individual_index, sparsity(noisy_sample),
                                                                      noisy_sample_performance))
            print("Constrain value:{}".format(constrain_value))

            current_gen_accuracies.append(noisy_sample_performance)
            current_gen_models.append(noisy_sample)
            current_gen_prs.append(pr_for_individual)

        best_index = np.argmax(current_gen_accuracies)
        gen_best_accuracy = current_gen_accuracies[best_index]
        if cfg.use_wandb:
            test_accuracy = test(current_gen_models[best_index], use_cuda, [get_random_batch(testloader)], verbose=0,
                                 one_batch=cfg.one_batch)
            log_dict = {"val_set_accuracy": gen_best_accuracy, "generation": gen,
                        "sparsity": sparsity(current_gen_models[best_index]),
                        "Deterministic performance": deterministic_pruning_performance,
                        "sparse_flops": total_sparse_flops,
                        "test_set_accuracy": test_accuracy
                        }
            log_dict.update(current_gen_prs[best_index])
            wandb.log(log_dict)
        if gen_best_accuracy >= best_accuracy_found:
            best_accuracy_found = gen_best_accuracy
            best_model_found = current_gen_models[best_index]
            best_pruning_rates_found = current_gen_prs[best_index]

            ### I don't want the pruning to be iterative at this stage
            ### so I remove the parametrization so the prune_with_rate
            ### method do not prune over the mask that is found

            temp = copy.deepcopy(best_model_found)
            remove_reparametrization(temp, exclude_layer_list=cfg.exclude_layers)
            current_best_model = temp

        print("Current best accuracy: {}".format(best_accuracy_found))
    # Test the best model found in the test set

    performance_best_model_found = test(best_model_found, use_cuda, testloader, verbose=1)
    print("The performance of on test set the best model is {} with pruning rate of {} and actual sparsity of {} "
          "".format(
        performance_best_model_found, cfg.amount, sparsity(best_model_found)))
    print("Performance of deterministic pruning is: {}".format(deterministic_pruning_performance))
    print("\n Best trial:")
    trial = study.best_trials[0]

    print("  Performance on validation set for set of sigmas: {}".format(trial.values))

    print("Set of sigmas and pruning rates: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    accuracy_string = "{:10.2f}".format(performance_best_model_found)
    pruning_rate_string = "{:0.3f}".format(sparsity(best_model_found))
    result = time.localtime(time.time())
    ######   Saving the model the pruning
    with open(cfg.save_model_path + "stochastic_pruning_pr_optim_{}_test_accuracy={}_time_{}-{}.pth".format(
            pruning_rate_string, accuracy_string, result.tm_hour, result.tm_min).replace(" ", ""),
              "wb") as f:
        pickle.dump(best_model_found, f)
    auxiliary_list_1 = list(best_pruning_rates_found.values())
    auxiliary_list_2 = list(best_pruning_rates_found.keys())

    dict_1 = {"layer": auxiliary_list_2, "pruning_rate": auxiliary_list_1}
    best_pr_csv = pd.DataFrame(dict_1)
    best_pr_csv.to_csv(cfg.save_data_path + "pruning_rate_per_layers_pr_optim_time_{}-{}.csv".format(result.tm_hour,
                                                                                                     result.tm_min),
                       sep=",", index=False)
    # with open(cfg.save_data_path + "pruning_rate_per_layers_pr_optim_time_{}-{}.pth".format(result.tm_hour,
    #                                                                                         result.tm_min),
    #           "wb") as f:
    #     pickle.dump(best_pruning_rates_found, f)
    #
    if cfg.use_wandb:
        wandb.join()


def generation_of_stochastic_prune_with_efficient_evaluation(solution, target_sparsity, sigmas_for_experiment,
                                                             population, dataloader, image_flops, total_flops, cfg,
                                                             previews_dict_of_image: dict = None):
    surviving_models = []
    first_iteration = 1
    image, y = None, None
    sorted_images = []
    dict_of_images = None
    if previews_dict_of_image is None:
        image, y = get_random_image_label(dataloader)
        dict_of_images = {}
    else:
        sorted_percentages = list(previews_dict_of_image.keys())
        sorted_percentages.sort(reverse=True)
        for percentage in sorted_percentages:
            sorted_images.append(previews_dict_of_image[percentage])
        image, y = sorted_images.pop()
    while len(surviving_models) == 0:
        for i in range(population):
            individual = copy.deepcopy(solution)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_experiment)
            ############################## we prune #######################################
            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)

            #######################################################33
            # remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)

            prediction = torch.argmax(noisy_sample(image).detach())
            total_flops += image_flops
            if prediction.eq(y.data):
                surviving_models.append(noisy_sample)

        if len(surviving_models) == 0:
            image, y = get_random_image_label(dataloader)

    if len(sorted_images) == 0 and previews_dict_of_image is None:
        dict_of_images[1 - len(surviving_models) / population] = (image, y)
        image, y = get_random_image_label(dataloader)
    if len(sorted_images) == 0 and previews_dict_of_image is not None:
        if len(previews_dict_of_image.keys()) == 1:
            image, y = get_random_image_label(dataloader)
        # else:
        #     previews_dict_of_image[1-len(surviving_models)/population] = (image,y)
        #     image, y = get_random_image_label(dataloader)
    if len(sorted_images) != 0:
        image, y = sorted_images.pop()
    index_to_remove: List[int] = []
    ######## While there
    total_for_image = len(surviving_models)
    while len(surviving_models) > 1:
        for ind in surviving_models:
            prediction = torch.argmax(ind(image).detach())
            total_flops += image_flops
            if not prediction.eq(y.data):
                index_to_remove.append(surviving_models.index(ind))

        if len(index_to_remove) == len(surviving_models):
            if len(sorted_images) == 0:
                image, y = get_random_image_label(dataloader)
            else:
                image, y = sorted_images.pop()
            index_to_remove = []

        else:
            if len(sorted_images) == 0 and previews_dict_of_image is not None:
                previews_dict_of_image[1 - len(surviving_models) / total_for_image] = (image, y)
            if len(sorted_images) == 0 and previews_dict_of_image is None:
                dict_of_images[1 - len(surviving_models) / population] = (image, y)
            for indx in index_to_remove:
                surviving_models.pop(indx)
            total_for_image = len(surviving_models)
    if previews_dict_of_image is None:
        return surviving_models[0], dict_of_images
    else:
        return surviving_models[0]


def efficient_evaluation_random_images(solution, target_sparsity, sigmas_for_experiment,
                                       population, dataloader, image_flops, total_flops, cfg,
                                       previews_dict_of_image: dict = None):
    surviving_models = []

    image, y = get_random_image_label(dataloader)
    image, y = image.cuda(), y.cuda()
    images_used = 1
    solution.cuda()
    print("First loop evaluation")
    while len(surviving_models) == 0:
        for i in range(population):

            individual = copy.deepcopy(solution)

            noisy_sample = get_noisy_sample_sigma_per_layer(individual, cfg, sigmas_for_experiment)
            ############################## we prune #######################################
            if cfg.pruner == "global":
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="global")
            else:
                prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers,
                                type="layer-wise",
                                pruner=cfg.pruner)

            #######################################################33
            # remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)

            prediction = torch.argmax(noisy_sample(image).detach())
            total_flops += image_flops
            if prediction.eq(y.data):
                surviving_models.append(noisy_sample)

        if len(surviving_models) == 0:
            image, y = get_random_image_label(dataloader)
            images_used += 1
            image, y = image.cuda(), y.cuda()

    print(f"At the end {len(surviving_models)} individuals survived the first loop out of {population} using "
          f"{images_used} images")
    image, y = get_random_image_label(dataloader)
    images_used += 1
    image, y = image.cuda(), y.cuda()

    index_to_remove: List[int] = []
    ######## While there
    total_for_image = len(surviving_models)
    while len(surviving_models) > 1:
        for index, ind in enumerate(surviving_models):
            prediction = torch.argmax(ind(image).detach())
            total_flops += image_flops
            if not prediction.eq(y.data):
                index_to_remove.append(index)
        print("Index to remove {} {}".format(len(index_to_remove), index_to_remove))
        print("Surviving Models {}".format(len(surviving_models)))
        if len(index_to_remove) >= len(surviving_models):
            image, y = get_random_image_label(dataloader)
            images_used += 1
            image, y = image.cuda(), y.cuda()
            index_to_remove = []

        if len(index_to_remove) < len(surviving_models):
            for index in index_to_remove:
                surviving_models.pop(index)

            index_to_remove = []
    print(f"Used in total {images_used} images")
    return surviving_models[0]


def fine_tune_after_stochatic_pruning_experiment(cfg: omegaconf.DictConfig, print_exclude_layers=True):
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()

    ################################## WANDB configuration ############################################
    exclude_layers_string = "_exclude_layers_fine_tuned" if cfg.fine_tune_exclude_layers else ""
    non_zero_string = "_non_zero_weights_fine_tuned" if cfg.fine_tune_non_zero_weights else ""
    one_batch_string = "_one_batch_per_generation" if cfg.one_batch else "_whole_dataset_per_generation"
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"fine_tune_base_stochastic_pruning_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}"
                 f"{non_zero_string}{one_batch_string}",
            notes="This run, run one iteration of stochastic pruning with fine tune on top of that. I'm doing lamp "
                  "with the pruning rates calculated no the original model and record the pruning rates of the noisy "
                  "models to compare",
            reinit=True,
        )
       ################################## Gradient flow measure############################################
    filepath_GF_measure =""
    if cfg.measure_gradient_flow:

        identifier = f"{time.time():14.2f}".replace(" ", "")
        if cfg.pruner == "lamp":
            filepath_GF_measure += "gradient_flow_data/stochastic_LAMP/{}/sigma{}/pr{}/{}/".format(cfg.architecture,cfg.sigma,cfg.amount,identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
                # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
            # else:
            # filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
        if cfg.pruner == "global":
            filepath_GF_measure += "gradient_flow_data/stochastic_GLOBAL/{}/sigma{}/pr{}/{}/".format(cfg.architecture,cfg.sigma,cfg.amount,identifier)
            path: Path = Path(filepath_GF_measure)
            if not path.is_dir():
                path.mkdir(parents=True)
            # else:
            #     filepath_GF_measure+=  f"fine_tune_pr_{cfg.amount}{exclude_layers_string}{non_zero_string}"
    pruned_model = get_model(cfg)
    best_model = None
    best_accuracy = 0
    initial_flops = 0
    data_loader_iterator = cycle(iter(valloader))
    data, y = next(data_loader_iterator)
    first_iter = 1
    unit_sparse_flops = 0
    evaluation_set = valloader
    if cfg.one_batch:
        evaluation_set = [(data, y)]
    names, weights = zip(*get_layer_dict(pruned_model))
    sigma_per_layer = dict(zip(names, [cfg.sigma] * len(names)))

    pr_per_layer = prune_with_rate(copy.deepcopy(pruned_model), target_sparsity, exclude_layers=cfg.exclude_layers,
                                   type="layer-wise",
                                   pruner="lamp", return_pr_per_layer=True)
    if cfg.use_wandb:
        log_dict = {}
        for name, elem in pr_per_layer.items():
            log_dict["deterministic_{}_pr".format(name)] = elem
        wandb.log(log_dict)
    for n in range(cfg.population):
        # current_model = get_noisy_sample(pruned_model, cfg)
        current_model = get_noisy_sample_sigma_per_layer(pruned_model, cfg, sigma_per_layer)
        copy_of_pruned_model = copy.deepcopy(current_model)
        # det_mask_transfer_model = copy.deepcopy(current_model)
        # copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        # det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)

        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        # Dense stochastic performance
        if cfg.pruner == "global":
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")

        if cfg.pruner == "manual":
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner="manual", pr_per_layer=pr_per_layer)
            individual_prs_per_layer = prune_with_rate(copy_of_pruned_model, target_sparsity,
                                                       exclude_layers=cfg.exclude_layers, type="layer-wise",
                                                       pruner="lamp", return_pr_per_layer=True)

            if cfg.use_wandb:
                log_dict = {}
                for name, elem in individual_prs_per_layer.items():
                    log_dict["individual_{}_pr".format(name)] = elem
                wandb.log(log_dict)

        if cfg.pruner == "lamp":
            prune_with_rate(current_model, target_sparsity, exclude_layers=cfg.exclude_layers,
                            type="layer-wise",
                            pruner=cfg.pruner)
        # prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                 pruner=cfg.pruner)

        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
        if first_iter:
            _, unit_sparse_flops = flops(current_model, data)
            first_iter = 0
        noisy_sample_performance, individual_sparse_flops = test(current_model, use_cuda, evaluation_set, verbose=0,
                                                                 count_flops=True, batch_flops=unit_sparse_flops)
        check_for_layers_collapse(current_model)
        initial_flops += individual_sparse_flops
        if noisy_sample_performance > best_accuracy:
            best_accuracy = noisy_sample_performance
            best_model = current_model

    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)
    initial_performance = test(best_model, use_cuda=use_cuda, testloader=valloader, verbose=1)
    initial_test_performance = test(best_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"val_set_accuracy": initial_performance, "sparse_flops": initial_flops, "initial_test_performance":
            initial_test_performance})


    restricted_fine_tune_measure_flops(best_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                       use_wandb=cfg.use_wandb, epochs=cfg.epochs, exclude_layers=cfg.exclude_layers,
                                       initial_flops=initial_flops,
                                       fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                       fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                       gradient_flow_file_prefix=filepath_GF_measure,
                                       cfg=cfg)



def one_shot_static_sigma_stochastic_pruning(cfg, eval_set="test", print_exclude_layers=True):
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    exclude_layers_string = "_exclude_layers" if print_exclude_layers else ""
    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"one_shot_stochastic_pruning_static_sigma_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}",
            notes="This experiment is to test if iterative global stochastic pruning, compares to one-shot stochastic pruning",
            reinit=True,
        )

    pruned_model = get_model(cfg)
    best_model = None
    best_accuracy = 0
    initial_flops = 0
    data_loader_iterator = cycle(iter(valloader))
    data, y = next(data_loader_iterator)
    first_iter = 1
    unit_sparse_flops = 0
    evaluation_set = None
    if cfg.one_batch:
        evaluation_set = [data]
    else:
        if eval_set == "test":
            evaluation_set = testloader
        if eval_set == "val":
            evaluation_set = valloader

    for n in range(cfg.population):

        noisy_sample = get_noisy_sample(pruned_model, cfg)

        # det_mask_transfer_model = copy.deepcopy(current_model)
        # copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        # det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)

        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        # Dense stochastic performance
        if cfg.pruner == "global":
            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
        else:
            prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner=cfg.pruner)
        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)
        if first_iter:
            _, unit_sparse_flops = flops(noisy_sample, data)
            first_iter = 0

        noisy_sample_performance, individual_sparse_flops = test(noisy_sample, use_cuda, evaluation_set, verbose=0,
                                                                 count_flops=True, batch_flops=unit_sparse_flops)

        initial_flops += individual_sparse_flops
        if cfg.use_wandb:
            test_accuracy = test(noisy_sample, use_cuda, [get_random_batch(testloader)], verbose=0)
            log_dict = {"val_set_accuracy": noisy_sample_performance, "individual": n,
                        "sparse_flops": initial_flops,
                        "test_set_accuracy": test_accuracy
                        }
            wandb.log(log_dict)
        if noisy_sample_performance > best_accuracy:
            best_accuracy = noisy_sample_performance
            best_model = noisy_sample
            test_accuracy = test(best_model, use_cuda, [get_random_batch(testloader)], verbose=0)
            if cfg.use_wandb:
                log_dict = {"best_val_set_accuracy": best_accuracy, "individual": n,
                            "sparsity": sparsity(best_model),
                            "sparse_flops": initial_flops,
                            "test_set_accuracy": test_accuracy
                            }
                wandb.log(log_dict)


def select_pruning(pruned_model, cfg, target_sparsity, use_stochastic, valloader, sigmas_for_experiment, image_flops,
                   total_flops,
                   dict_of_images=None):
    if use_stochastic:

        return efficient_evaluation_random_images(pruned_model, target_sparsity,
                                                  sigmas_for_experiment,
                                                  cfg.population, valloader,
                                                  image_flops,
                                                  total_flops=total_flops,
                                                  cfg=cfg,
                                                  previews_dict_of_image=dict_of_images)
        # if dict_of_images is None:
        #     best_model,dict_image = generation_of_stochastic_prune_with_efficient_evaluation(pruned_model,
        #                                                                                    target_sparsity,
        #                                                                           sigmas_for_experiment,
        #                                                                           cfg.population, valloader,
        #                                                                           image_flops,
        #                                                                           total_flops=total_flops,
        #                                                                           cfg=cfg,
        #                                                                           previews_dict_of_image=dict_of_images)
        #     return best_model,dict_image
        # else:
        #     best_model = generation_of_stochastic_prune_with_efficient_evaluation(pruned_model,
        #                                                                                       target_sparsity,
        #                                                                                       sigmas_for_experiment,
        #                                                                                       cfg.population, valloader,
        #                                                                                       image_flops,
        #                                                                                       total_flops=total_flops,
        #                                                                                       cfg=cfg,
        #                                                                                       previews_dict_of_image=dict_of_images)
        #     return best_model

    else:
        if cfg.pruner == "global":
            prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
        else:
            prune_with_rate(pruned_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner=cfg.pruner)
        return pruned_model


def lamp_scenario_2_cheap_evaluation(cfg):
    trainloader, valloader, testloader = get_cifar_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    original_epoch_number = cfg.epochs
    exclude_layers_string = "_exclude_layers_fine_tuned" if cfg.fine_tune_exclude_layers else ""
    non_zero_string = "_non_zero_weights_fine_tuned" if cfg.fine_tune_non_zero_weights else ""
    full_fine_tune_string = "_full_fine_tune" if cfg.full_fine_tune else "_single_step_fine_tune"
    stochastic_pruning_string = "_stochastic_pruning" if cfg.use_stochastic else "_deterministic_pruning"

    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"progressive_pruning_fine_tune_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}"
                 f"{non_zero_string}{full_fine_tune_string}{stochastic_pruning_string}",
            reinit=True,
        )
    pruned_model = get_model(cfg)
    for_flops_model = copy.deepcopy(pruned_model)
    if cfg.pruner == "global":
        prune_with_rate(for_flops_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(for_flops_model, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)
    sigmas_for_experiment = None
    names, weights = zip(*get_layer_dict(pruned_model))
    list_of_sigma = [cfg.sigma] * len(names)
    sigmas_for_experiment = dict(zip(names, list_of_sigma))

    # remove_reparametrization(model=pruned_model, exclude_layer_list=cfg.exclude_layers)
    image, y = get_random_image_label(valloader)
    _, image_flops = flops(for_flops_model, image)
    TOTAL_FLOPS = 0
    dict_of_images = None
    if cfg.use_stochastic:
        pruned_model = select_pruning(pruned_model, cfg=cfg, target_sparsity=target_sparsity,
                                      use_stochastic=cfg.use_stochastic,
                                      valloader=valloader, sigmas_for_experiment=sigmas_for_experiment,
                                      image_flops=image_flops,
                                      total_flops=TOTAL_FLOPS)
    else:
        pruned_model = select_pruning(pruned_model, cfg=cfg, target_sparsity=target_sparsity,
                                      use_stochastic=cfg.use_stochastic,
                                      valloader=valloader, sigmas_for_experiment=sigmas_for_experiment,
                                      image_flops=image_flops,
                                      total_flops=TOTAL_FLOPS)

    initial_performance = test(pruned_model, use_cuda=use_cuda, testloader=valloader, verbose=1)
    initial_performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
    if cfg.use_wandb:
        wandb.log({"val_set_accuracy": initial_performance, "sparse_flops": TOTAL_FLOPS,
                   "test_set_accuracy": initial_performance_test_set, "G": -1,
                   "sparsity": sparsity(model=pruned_model)})
    if cfg.full_fine_tune:

        TOTAL_FLOPS = restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                                         use_wandb=cfg.use_wandb, epochs=cfg.epochs,
                                                         exclude_layers=cfg.exclude_layers,
                                                         fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                         fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                         initial_flops=TOTAL_FLOPS)
        performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "test_set_accuracy": performance_test_set, "G": 0,
                       "sparsity": sparsity(model=pruned_model)})

        pruned_model = select_pruning(pruned_model, cfg=cfg, target_sparsity=0.5,
                                      use_stochastic=cfg.use_stochastic,
                                      valloader=valloader, sigmas_for_experiment=sigmas_for_experiment,
                                      image_flops=image_flops,
                                      total_flops=TOTAL_FLOPS, dict_of_images=dict_of_images)

        # if cfg.pruner == "global":
        #     prune_with_rate(pruned_model, 0.95, exclude_layers=cfg.exclude_layers, type="global")
        # else:
        #     prune_with_rate(pruned_model, 0.95, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                     pruner=cfg.pruner)

        TOTAL_FLOPS = restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                                         use_wandb=cfg.use_wandb, epochs=cfg.epochs,
                                                         exclude_layers=cfg.exclude_layers,
                                                         fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                         fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                         initial_flops=TOTAL_FLOPS)
        performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "test_set_accuracy": performance_test_set, "G": 1,
                       "sparsity": sparsity(model=pruned_model)})
        # if cfg.pruner == "global":
        #     prune_with_rate(pruned_model, 0.99, exclude_layers=cfg.exclude_layers, type="global")
        # else:
        #     prune_with_rate(pruned_model, 0.99, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                     pruner=cfg.pruner)
        pruned_model = select_pruning(pruned_model, cfg=cfg, target_sparsity=0.8,
                                      use_stochastic=cfg.use_stochastic,
                                      valloader=valloader, sigmas_for_experiment=sigmas_for_experiment,
                                      image_flops=image_flops,
                                      total_flops=TOTAL_FLOPS, dict_of_images=dict_of_images)

        TOTAL_FLOPS = restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                                         use_wandb=cfg.use_wandb, epochs=cfg.epochs,
                                                         exclude_layers=cfg.exclude_layers,
                                                         fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                         fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                         initial_flops=TOTAL_FLOPS)
        performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "test_set_accuracy": performance_test_set, "G": 2,
                       "sparsity": sparsity(model=pruned_model)})
    else:

        TOTAL_FLOPS = restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                                         use_wandb=cfg.use_wandb, epochs=cfg.short_epochs,
                                                         exclude_layers=cfg.exclude_layers,
                                                         fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                         fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                         initial_flops=TOTAL_FLOPS)
        performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "test_set_accuracy": performance_test_set, "G": 0,
                       "sparsity": sparsity(model=pruned_model)})
        pruned_model = select_pruning(pruned_model, cfg=cfg, target_sparsity=0.5,
                                      use_stochastic=cfg.use_stochastic,
                                      valloader=valloader, sigmas_for_experiment=sigmas_for_experiment,
                                      image_flops=image_flops,
                                      total_flops=TOTAL_FLOPS, dict_of_images=dict_of_images)
        performance_val_set = test(pruned_model, use_cuda=use_cuda, testloader=valloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "val_set_accuracy": performance_val_set,
                       "sparsity": sparsity(model=pruned_model)})
        # if cfg.pruner == "global":
        #     prune_with_rate(pruned_model, 0.95, exclude_layers=cfg.exclude_layers, type="global")
        # else:
        #     prune_with_rate(pruned_model, 0.95, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                     pruner=cfg.pruner)
        TOTAL_FLOPS = restricted_fine_tune_measure_flops(pruned_model, valloader, testloader,
                                                         FLOP_limit=cfg.flop_limit,
                                                         use_wandb=cfg.use_wandb, epochs=cfg.short_epochs,
                                                         exclude_layers=cfg.exclude_layers,
                                                         fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                         fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                         initial_flops=TOTAL_FLOPS)
        performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "test_set_accuracy": performance_test_set, "G": 1,
                       "sparsity": sparsity(model=pruned_model)})
        pruned_model = select_pruning(pruned_model, cfg=cfg, target_sparsity=0.8,
                                      use_stochastic=cfg.use_stochastic,
                                      valloader=valloader, sigmas_for_experiment=sigmas_for_experiment,
                                      image_flops=image_flops,
                                      total_flops=TOTAL_FLOPS, dict_of_images=dict_of_images)
        performance_val_set = test(pruned_model, use_cuda=use_cuda, testloader=valloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "val_set_accuracy": performance_val_set,
                       "sparsity": sparsity(model=pruned_model)})
        # if cfg.pruner == "global":
        #     prune_with_rate(pruned_model, 0.99, exclude_layers=cfg.exclude_layers, type="global")
        # else:
        #     prune_with_rate(pruned_model, 0.99, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                     pruner=cfg.pruner)
        # cfg.epoch = original_epoch_number
        TOTAL_FLOPS = restricted_fine_tune_measure_flops(pruned_model, valloader, testloader, FLOP_limit=cfg.flop_limit,
                                                         use_wandb=cfg.use_wandb, epochs=cfg.epochs,
                                                         exclude_layers=cfg.exclude_layers,
                                                         fine_tune_exclude_layers=cfg.fine_tune_exclude_layers,
                                                         fine_tune_non_zero_weights=cfg.fine_tune_non_zero_weights,
                                                         initial_flops=TOTAL_FLOPS)
        performance_test_set = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
        if cfg.use_wandb:
            wandb.log({"sparse_flops": TOTAL_FLOPS,
                       "test_set_accuracy": performance_test_set, "G": 2,
                       "sparsity": sparsity(model=pruned_model)})

    last_performance = test(pruned_model, use_cuda=use_cuda, testloader=testloader, verbose=1)
    accuracy_string = "{:10.2f}".format(last_performance).replace(" ", "")
    result = time.localtime(time.time())
    model_file_name = cfg.save_model_path + f"progressive_pruning_fine_tune_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}"
    f"{non_zero_string}{full_fine_tune_string}{stochastic_pruning_string}_test_accuracy={accuracy_string}_time_" \
    f"{result.tm_hour}-{result.tm_min}.pth"
    model_file_name = model_file_name.replace(" ", "")
    with open(model_file_name, "wb") as f:
        pickle.dump(best_model_found, f)


def experiment_selector(cfg: omegaconf.DictConfig, number_experiment: int = 1):
    if number_experiment == 1:
        dynamic_sigma_per_layer_one_shot_pruning(cfg)
    if number_experiment == 2:
        static_sigma_per_layer_manually_iterative_process(cfg)
    if number_experiment == 3:
        static_sigma_per_layer_optimized_iterative_process(cfg)
    if number_experiment == 4:
        static_global_sigma_iterative_process(cfg)
    if number_experiment == 5:
        dynamic_sigma_iterative_process(cfg)
    if number_experiment == 6:
        run_fine_tune_experiment(cfg)
    if number_experiment == 7:
        static_sigma_per_layer_manually_iterative_process_flops_counts(cfg)
    if number_experiment == 8:
        stochastic_pruning_with_static_sigma_and_pr_optimization(cfg)
    if number_experiment == 9:
        # one_shot_iterative_sotchastic_pruning(cfg)
        pass
    if number_experiment == 10:
        one_shot_static_sigma_stochastic_pruning(cfg, eval_set="val")
    if number_experiment == 11:
        fine_tune_after_stochatic_pruning_experiment(cfg)
    if number_experiment == 12:
        lamp_scenario_2_cheap_evaluation(cfg)


def pruning_rate_experiment_selector(cfg: omegaconf.DictConfig, number_experiment: int = 1):
    if number_experiment == 1:
        pass
    if number_experiment == 2:
        pass
    if number_experiment == 3:
        pass
    if number_experiment == 4:
        pass
    if number_experiment == 5:
        pass


def test_sigma_experiment_selector():
    test_cfg = omegaconf.DictConfig({
        "population": 2,
        "generations": 2,
        "architecture": "resnet18",
        "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
        "noise": "gaussian",
        "pruner": "erk",
        "type": "alternative",
        "exclude_layers": ["conv1", "linear"],
        # "sigma": 0.0021419609859022197,
        "sigma": 0.005,
        "amount": 0.9,
        "batch_size": 512,
        "num_workers": 0,
        "save_model_path": "stochastic_pruning_models/",
        "save_data_path": "stochastic_pruning_data/",
        "use_wandb": False
    })
    experiment_selector(test_cfg, 1)
    experiment_selector(test_cfg, 2)
    experiment_selector(test_cfg, 3)
    experiment_selector(test_cfg, 4)
    experiment_selector(test_cfg, 5)


############################### Plot of stochastic pruning against deterministic pruning ###############################

def stochastic_pruning_against_deterministic_pruning(cfg: omegaconf.DictConfig, eval_set: str = "test"):
    use_cuda = torch.cuda.is_available()
    net = get_model(cfg)
    evaluation_set = select_eval_set(cfg, eval_set)
    N = cfg.population
    pop = []
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []
    original_performance = test(net, use_cuda, evaluation_set, verbose=1)
    pruned_original = copy.deepcopy(net)
    prune_with_rate(pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers)
    # remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(pruned_original, use_cuda, evaluation_set, verbose=1)
    pop.append(pruned_original)
    pruned_performance.append(pruned_original_performance)
    labels = ["pruned original"]
    stochastic_dense_performances.append(original_performance)
    for n in range(N):
        current_model = get_noisy_sample(net, cfg)

        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        print("Stochastic dense performance")
        StoDense_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        # Dense stochastic performance
        stochastic_dense_performances.append(StoDense_performance)

        prune_with_rate(current_model, amount=cfg.amount, exclude_layers=cfg.exclude_layers)

        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)

        stochastic_pruned_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        pruned_performance.append(stochastic_pruned_performance)
        stochastic_deltas.append(StoDense_performance - stochastic_pruned_performance)

    # len(pruned performance)-1 because the first one is the pruned original
    labels.extend(["stochastic pruned"] * (len(pruned_performance) - 1))

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

    del pop
    cutoff = original_performance - 2
    ################################# plotting the comparison #########################################################
    fig, ax = plt.subplots()

    original_line = ax.axhline(y=original_performance, color="k", linestyle="-", label="Original Performance")

    deterministic_pruning_line = ax.axhline(y=pruned_original_performance, c="purple", label="Deterministic Pruning")
    plt.xlabel("Ranking Index", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    stochastic_models_points_dense = []
    stochastic_models_points_pruned = []
    transfer_mask_models_points = []
    stochastic_with_deterministic_mask_models_points = []

    for i, element in enumerate(pruned_performance[ranked_index]):
        if i == index_of_pruned_original:
            assert element == pruned_original_performance, "The supposed pruned original is not the original: element " \
                                                           f"in list {element} VS pruned performance:" \
                                                           f" {pruned_original_performance}"
            # p1 = ax.scatter(i, element, c="g", marker="o")
        else:
            if labels[ranked_index[i]] == "sto mask transfer":
                sto_transfer_point = ax.scatter(i, element, c="tab:orange", marker="P")
                transfer_mask_models_points.append(sto_transfer_point)
            elif labels[ranked_index[i]] == "det mask transfer":
                det_transfer_point = ax.scatter(i, element, c="tab:olive", marker="X")
                stochastic_with_deterministic_mask_models_points.append(det_transfer_point)
            else:
                pruned_point = ax.scatter(i, element, c="steelblue", marker="x")
                stochastic_models_points_pruned.append(pruned_point)
    for i, element in enumerate(stochastic_dense_performances[ranked_index]):
        if i == index_of_pruned_original or element == 1:
            continue
            # ax.scatter(i, element, c="y", marker="o", label="original model performance")
        else:
            dense_point = ax.scatter(i, element, c="c", marker="1")
            stochastic_models_points_dense.append(dense_point)

    plt.legend([original_line, tuple(stochastic_models_points_pruned), tuple(stochastic_models_points_dense),
                deterministic_pruning_line],
               ['Original Performance', 'Pruned Stochastic', 'Dense Stochastic', "Deterministic Pruning"],
               scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)})
    plt.ylim(20, 100)
    ax2 = ax.twinx()

    epsilon_ticks = original_performance - np.linspace(20, 100, 9)
    ax2.set_yticks(epsilon_ticks, minor=False)
    ax2.set_ylabel(r"$\epsilon$", fontsize=20)
    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis="y", colors="red")
    ax2.yaxis.label.set_color('red')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        f"data/figures/stochastic_deterministic_{cfg.noise}_sigma_"
        f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
        f"_{cfg.population}_{eval_set}.pdf")
    plt.savefig(f"data/figures/stochastic_deterministic_{cfg.noise}_sigma_"
                f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
                f"_{cfg.population}_{eval_set}.png")


# def stochastic_pruning_global_against_LAMP_deterministic_pruning():
def stochastic_pruning_global_against_LAMP_deterministic_pruning(cfg: omegaconf.DictConfig, eval_set: str = "test"):
    use_cuda = torch.cuda.is_available()
    net = get_model(cfg)
    evaluation_set = select_eval_set(cfg, eval_set)
    N = cfg.population
    pop = []
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []
    deterministic_with_stochastic_mask_performance = []
    stochastic_with_deterministic_mask_performance = []
    original_performance = test(net, use_cuda, evaluation_set, verbose=1)

    lamp_pruned_original: Union[Union[ResNet, None, VGG], Any] = copy.deepcopy(net)

    prune_with_rate(lamp_pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                    pruner="lamp")
    # remove_reparametrization(pruned_original)
    print("pruned_performance of pruned original")
    pruned_original_performance = test(lamp_pruned_original, use_cuda, evaluation_set, verbose=1)
    pop.append(lamp_pruned_original)
    pruned_performance.append(pruned_original_performance)
    labels = ["pruned original"]
    stochastic_dense_performances.append(original_performance)
    for n in range(N):
        # sto_mask_transfer_model = copy.deepcopy(net)

        current_model = get_noisy_sample(net, cfg)

        # det_mask_transfer_model = copy.deepcopy(current_model)
        # copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        # det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)

        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        print("Stochastic dense performance")
        StoDense_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        # Dense stochastic performance
        stochastic_dense_performances.append(StoDense_performance)

        prune_with_rate(current_model, amount=cfg.amount, exclude_layers=cfg.exclude_layers)

        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)

        stochastic_pruned_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        pruned_performance.append(stochastic_pruned_performance)
        stochastic_deltas.append(StoDense_performance - stochastic_pruned_performance)

    # len(pruned performance)-1 because the first one is the pruned original
    labels.extend(["stochastic pruned"] * (len(pruned_performance) - 1))
    # labels.extend(["sto mask transfer"] * len(deterministic_with_stochastic_mask_performance))
    # pruned_performance.extend(deterministic_with_stochastic_mask_performance)
    # stochastic_dense_performances.extend([1] * len(deterministic_with_stochastic_mask_performance))
    # # Deterministic mask transfer to stochastic model
    # labels.extend(["det mask transfer"] * len(stochastic_with_deterministic_mask_performance))
    # pruned_performance.extend(stochastic_with_deterministic_mask_performance)
    # stochastic_dense_performances.extend([1] * len(stochastic_with_deterministic_mask_performance))

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

    del pop
    cutoff = original_performance - 2
    ################################# plotting the comparison #########################################################
    fig, ax = plt.subplots()

    original_line = ax.axhline(y=original_performance, color="k", linestyle="-", label="Original Performance")

    deterministic_pruning_line = ax.axhline(y=pruned_original_performance, c="purple", label="Deterministic Pruning")
    # plt.axhline(y=cutoff, color="r", linestyle="--", label="cutoff value")
    plt.xlabel("Ranking Index", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    sigma = format_sigmas(cfg.sigma)
    pr = format_percentages(cfg.amount)
    # if cfg.noise == "geogaussian":
    #     plt.title(f"CIFAR10 Geometric Gaussian Noise $\sigma={sigma},pr={pr}$", fontsize=10)
    # if cfg.noise == "gaussian":
    #     plt.title(f"CIFAR10 Additive Gaussian Noise $\sigma={sigma},pr={pr}$", fontsize=10)

    stochastic_models_points_dense = []
    stochastic_models_points_pruned = []
    transfer_mask_models_points = []
    stochastic_with_deterministic_mask_models_points = []

    p1 = None

    for i, element in enumerate(pruned_performance[ranked_index]):
        # if i == index_of_pruned_original:
        #     assert element == pruned_original_performance, "The supposed pruned original is not the original: element " \
        #                                                    f"in list {element} VS pruned performance:" \
        #                                                    f" {pruned_original_performance}"
        # p1 = ax.scatter(i, element, c="g", marker="o")
        # else:
        if labels[ranked_index[i]] == "sto mask transfer":
            sto_transfer_point = ax.scatter(i, element, c="tab:orange", marker="P")
            transfer_mask_models_points.append(sto_transfer_point)
        elif labels[ranked_index[i]] == "det mask transfer":
            det_transfer_point = ax.scatter(i, element, c="tab:olive", marker="X")
            stochastic_with_deterministic_mask_models_points.append(det_transfer_point)
        else:
            pruned_point = ax.scatter(i, element, c="steelblue", marker="x")
            stochastic_models_points_pruned.append(pruned_point)
    for i, element in enumerate(stochastic_dense_performances[ranked_index]):
        if i == index_of_pruned_original or element == 1:
            continue
            # ax.scatter(i, element, c="y", marker="o", label="original model performance")
        else:
            dense_point = ax.scatter(i, element, c="c", marker="1")
            stochastic_models_points_dense.append(dense_point)

    plt.legend([original_line, tuple(stochastic_models_points_pruned), tuple(stochastic_models_points_dense),
                deterministic_pruning_line],
               ['Original Performance', 'Pruned Stochastic', 'Dense Stochastic', "LAMP deterministic Pruned"],
               scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)})
    plt.ylim(20, 100)
    ax2 = ax.twinx()
    # y1_tick_positions = ax.get_ticks()
    epsilon_ticks = original_performance - np.linspace(20, 100, 9)
    ax2.set_yticks(epsilon_ticks, minor=False)
    ax2.set_ylabel(r"$\epsilon$", fontsize=20)
    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis="y", colors="red")
    ax2.yaxis.label.set_color('red')
    ax2.invert_yaxis()
    # ax2 = ax.twinx()
    # y1_tick_positions = ax.get_ticks()
    # epsilon_ticks = original_performance - np.linspace(88,100, 7)
    # ax2.set_yticks(np.flip(epsilon_ticks), minor=False)
    # ax2.tick_params(axis="y", colors="red")

    plt.tight_layout()
    plt.savefig(
        f"data/figures/LAMP_stochastic_{cfg.noise}_sigma_"
        f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
        f"_{cfg.population}_{eval_set}_{cfg.architecture}.pdf")
    plt.savefig(f"data/figures/LAMP_stochastic_{cfg.noise}_sigma_"
                f"{cfg.sigma}_pr_{cfg.amount}_batchSize_{cfg.batch_size}_pop"
                f"_{cfg.population}_{eval_set}_{cfg.architecture}.png")


############################# Iterative stochastic pruning #############################################################
def gradient_flow_correlation_analysis(prefix:str,cfg):
    # prefix = Path(prefix)

    deterministic_lamp_root = prefix + "deterministic_LAMP/" + f"{cfg.architecture}/{cfg.amount}/"

    deterministic_global_root = prefix + "deterministic_GLOBAL/" + f"{cfg.architecture}/{cfg.amount}/"

    stochastic_global_root = prefix + "stochastic_GLOBAL/" + f"{cfg.architecture}/sigma{cfg.sigma}/pr{cfg.amount}/"

    stochastic_lamp_root = prefix + "stochastic_LAMP/" + f"{cfg.architecture}/sigma{cfg.sigma}/pr{cfg.amount}/"


    combine_stochastic_GLOBAL_DF: pd.DataFrame = None
    combine_stochastic_LAMP_DF: pd.DataFrame = None
    combine_deterministic_GOBAL_DF: pd.DataFrame = None
    combine_deterministic_LAMP_DF: pd.DataFrame = None

    ########################### Global Determinisitc ########################################

    for index, individual in enumerate(glob.glob(deterministic_global_root+"*/",recursive=True)):
        individual_df = pd.read_csv(individual+"recordings.csv" ,sep=",",header=0,index_col=False)
        len_df = individual_df.shape[0]
        individual_df["individual"] = [index] * len_df
        individual_df["sigma"] = [cfg.sigma] * len_df
        if combine_deterministic_GOBAL_DF is None:
            combine_deterministic_GOBAL_DF = individual_df
        else:
            combine_deterministic_GOBAL_DF = pd.concat((combine_deterministic_GOBAL_DF,individual_df),ignore_index=True)

    combine_deterministic_GOBAL_DF.to_csv(f"gradientflow_deterministic_sigma{cfg.sigma}_global.csv",header=True,index=False)

    ########################### Lamp Deterministic  ########################################

    for index, individual in enumerate(glob.glob(deterministic_lamp_root+"*/",recursive=True)):
        individual_df = pd.read_csv(individual+"recordings.csv" ,sep=",",header=0,index_col=False)
        len_df = individual_df.shape[0]
        individual_df["individual"] = [index] * len_df
        individual_df["sigma"] = [cfg.sigma] * len_df
        if combine_deterministic_LAMP_DF is None:
            combine_deterministic_LAMP_DF = individual_df
        else:
            combine_deterministic_LAMP_DF  = pd.concat((combine_deterministic_LAMP_DF,individual_df),ignore_index=True)

    combine_deterministic_LAMP_DF.to_csv(f"gradientflow_deterministic_sigma{cfg.sigma}_lamp.csv",header=True,index=False)

    ########################## first Global stochatic #######################################
    for index, individual in enumerate(glob.glob(stochastic_global_root + "*/",recursive=True)):
        individual_df = pd.read_csv(individual +"recordings.csv" ,sep=",",header=0,index_col=False)
        len_df = individual_df.shape[0]
        individual_df["individual"] = [index] * len_df
        individual_df["sigma"] = [cfg.sigma] * len_df
        if combine_stochastic_GLOBAL_DF is None:
            combine_stochastic_GLOBAL_DF = individual_df
        else:
            combine_stochastic_GLOBAL_DF = pd.concat((combine_stochastic_GLOBAL_DF,individual_df),ignore_index=True)

    combine_stochastic_GLOBAL_DF.to_csv(f"gradientflow_stochastic_sigma{cfg.sigma}_global.csv",header=True,index=False)
    ########################## Second LAMP stochatic #######################################



    for index, individual in enumerate(glob.glob(stochastic_lamp_root+"*/",recursive=True)):
        individual_df = pd.read_csv(individual+"recordings.csv" ,sep=",",header=0,index_col=False)
        len_df = individual_df.shape[0]
        individual_df["individual"] = [index] * len_df
        individual_df["sigma"] = [cfg.sigma] * len_df
        if combine_stochastic_LAMP_DF is None:
            combine_stochastic_LAMP_DF = individual_df
        else:
            combine_stochastic_LAMP_DF  = pd.concat((combine_stochastic_LAMP_DF,individual_df),ignore_index=True)

    combine_stochastic_LAMP_DF.to_csv(f"gradientflow_stochastic_sigma{cfg.sigma}_lamp.csv",header=True ,index=False)
def plot_gradientFlow_data(filepath,title=""):
    data_frame= pd.read_csv(filepath,sep=",",header=0,index_col=False)

    sns.set_theme(style="dark")

    # Plot each year's time series in its own facet
    g = sns.relplot(
        data=data_frame,
        x="Epoch", y="test_set_gradient_magnitude", col="individual", hue="individual",
        kind="line", palette="crest", linewidth=4, zorder=5,
        col_wrap=10, height=2, aspect=1.5, legend=False,
    )

    # Iterate over each subplot to customize further
    for year, ax in g.axes_dict.items():
        # Add the title as an annotation within the plot
        ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")

        # Plot every year's time series in the background
        sns.lineplot(
            data=data_frame, x="Epoch", y="val_set_gradient_magnitude", units="individual",
            estimator=None, color=".7", linewidth=1, ax=ax,
        )

    # Reduce the frequency of the x axis ticks
    ax.set_xticks(ax.get_xticks()[::2])

    # Tweak the supporting aspects of the plot
    g.set_titles("")
    g.set_axis_labels("", r"$|\nabla\mathcal{L}|$")
    g.tight_layout()
    plt.title(title,fontsize=20)
    plt.savefig("le_test.png")

def get_first_epoch_GF_last_epoch_accuracy(dataFrame,title,file):
    initial_val_set_GF = []
    initial_test_set_GF = []
    final_test_performance = []
    initial_test_performance = []
    final_val_performance = []
    initial_val_performance = []
    average_test_improvement_rate = []
    average_val_improvement_rate = []
    for elem in dataFrame["individual"].unique():
        temp_df = dataFrame[dataFrame["individual"]==elem]
        initial_val_set_GF.append(float(temp_df['val_set_gradient_magnintude'][temp_df["Epoch"]==-1]))
        initial_test_set_GF.append(float(temp_df['test_set_gradient_magnitude'][temp_df["Epoch"]==-1]))
        final_test_performance.append(float(temp_df["test_accuracy"][temp_df["Epoch"]==90]))
        initial_test_performance.append(float(temp_df["test_accuracy"][temp_df["Epoch"]==-1]))
        final_val_performance.append(float(temp_df["val_accuracy"][temp_df["Epoch"]==90]))
        initial_val_performance.append(float(temp_df["val_accuracy"][temp_df["Epoch"]==-1]))
        test_difference = temp_df["test_accuracy"][1:].diff()
        val_difference = temp_df["val_accuracy"][1:].diff()
        average_test_improvement_rate.append(float(test_difference.mean()))
        average_val_improvement_rate.append(float(val_difference.mean()))
    d = pd.DataFrame({"initial_GF_valset":initial_val_set_GF,"initial_GF_testset":initial_test_set_GF,
                      "final_test_accuracy":final_test_performance,
                      "test_improvement_rate":average_test_improvement_rate,
                      "val_improvement_rate":average_val_improvement_rate,
                      "initial_test_accuracy":initial_test_performance,
                      "final_val_accuracy": final_val_performance,
                      "initial_val_accuracy": initial_val_performance,
                      })





    plt.figure()

    g = sns.scatterplot(data=d,x="initial_GF_valset",y="final_test_accuracy")
    # g.set_titles("")
    # g.set_axis_labels("", r"$|\nabla\mathcal{L}|$")
    plt.tight_layout()
    plt.title(title,fontsize=20)
    plt.savefig(f"{file}1.png", bbox_inches="tight")
    plt.figure()
    g = sns.scatterplot(data=d,x="initial_GF_testset",y="final_test_accuracy")
    plt.tight_layout()
    plt.title(title,fontsize=20)
    plt.savefig(f"{file}2.png", bbox_inches="tight")

    plt.figure()

    g = sns.scatterplot(data=d,x="initial_GF_valset",y="test_improvement_rate")
    plt.tight_layout()
    plt.title(title,fontsize=20)
    plt.savefig(f"{file}3.png", bbox_inches="tight")


    plt.figure()

    g = sns.scatterplot(data=d,x="initial_GF_valset",y="val_improvement_rate")
    plt.tight_layout()
    plt.title(title,fontsize=20)
    plt.savefig(f"{file}4.png", bbox_inches="tight")
    plt.figure()

    g = sns.scatterplot(data=d,x="initial_GF_valset",y="initial_test_accuracy")
    plt.tight_layout()
    plt.title(title,fontsize=20)
    plt.savefig(f"{file}5.png", bbox_inches="tight")

    plt.figure()

    g = sns.scatterplot(data=d,x="initial_GF_valset",y="initial_val_accuracy")
    plt.tight_layout()
    plt.title(title,fontsize=20)
    plt.savefig(f"{file}6.png", bbox_inches="tight")

    plt.figure()

    g = sns.scatterplot(data=d,x="initial_GF_valset",y="final_val_accuracy")
    plt.tight_layout()
    plt.title(title,fontsize=20)
    plt.savefig(f"{file}7.png", bbox_inches="tight")

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
    cfg = omegaconf.DictConfig({
        "population": 1,
        "generations": 10,
        "epochs": 100,
        "short_epochs": 10,
        # "architecture": "VGG19",
        "architecture": "resnet18",
        "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
         # "solution": "trained_models/cifar10/VGG19_cifar10_traditional_train_valacc=93,57.pth",
       "noise": "gaussian",
       "pruner": "lamp",
        "model_type": "alternative",
        "exclude_layers": ["conv1", "linear"],
        # "exclude_layers": ["features.0", "classifier"],
        "fine_tune_exclude_layers": True,
        "fine_tune_non_zero_weights": True,
        "sampler": "tpe",
        "flop_limit": 0,
        "one_batch": True,
        "measure_gradient_flow":True,
        "full_fine_tune": False,
        "use_stochastic": True,
        # "sigma": 0.0021419609859022197,
        "sigma": 0.0065,
        "noise_after_pruning":0,
        "amount": 0.9,
        "dataset": "cifar10",
        "batch_size": 512,
        # "batch_size": 128,
        "num_workers": 0,
        "save_model_path": "stochastic_pruning_models/",
        "save_data_path": "stochastic_pruning_data/",
        "use_wandb": True
    })

    # plot_val_accuracy_wandb("val_accuracy_iterative_erk_pr_0.9_sigma_manual_10_percentile_30-12-2022-.csv",
    #                         "val_acc_plot.pdf",
    #                         "generation", "iterative_erk_pr_0.9_sigma_manual_10_percentile - val_set_accuracy",
    #                         "Generations", "Accuracy")
    # task_run.shfilepaths = ["stochastic_global_fine_tuning_testset.csv","stochastic_lamp_fine_tuning_testset.csv"]
    # legends = ["Stochatic Global Pruning","Stochastic LAMP Pruning"]
    # filepaths = ["stochastic_global_fine_tuning_testset.csv"]
    # legends = ["Stochatic Global Pruning"]
    # plot_test_accuracy_wandb(filepaths,legends, "stochastic_pruning_fine_tuning_testset.pdf",
    #                          "sparse_flops", "architecture: resnet18 - test_set_accuracy", "architecture: resnet18 - "
    #                                                                                        "test_set_accuracy__MIN",
    #                          "architecture: resnet18 - test_set_accuracy__MAX", "FLOPS", "Tet set accuracy", "Accuracy")
    #

    # test_sigma_experiment_selector()
    # experiment_selector(cfg, 4)
    # experiment_selector(cfg, 6)
    # experiment_selector(cfg,6)
    # gradient_flow_correlation_analysis("gradient_flow_data/",cfg)
    # plot_gradientFlow_data("gradientflow_stochastic_global.csv","Global Stochastic")
    # stochastic_lamp_df = pd.read_csv("gradientflow_stochastic_lamp.csv", header=0, index_col=False)
    # stochastic_global_df = pd.read_csv("gradientflow_stochastic_global.csv", header=0, index_col=False)
    # get_first_epoch_GF_last_epoch_accuracy(stochastic_global_df,"Global Stochastic","images_global_stochastic")
    # get_first_epoch_GF_last_epoch_accuracy(stochastic_lamp_df,"Lamp Stochastic","images_lamp_stochastic")
    experiment_selector(cfg, 11)

    # stochastic_pruning_global_against_LAMP_deterministic_pruning(cfg)

    # stochastic_pruning_against_deterministic_pruning(cfg)
    # cfg.sigma = 0.002
    # cfg.amount = 0.8
    # stochastic_pruning_against_deterministic_pruning(cfg)

    ########### Mask Transfer experiments #############################
    # transfer_mask_rank_experiments(cfg)
    ############ Stchastic pruning VS Deterministic Pruning ###########
    # stochastic_pruning_against_deterministic_pruning(cfg)

    ########### All epsilon stochastic pruning #######################
    # fp = "data/epsilon_experiments_t_1-33_full.csv"
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.003],
    #                                           specific_pruning_rates=[0.9])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.003],
    #                                           specific_pruning_rates=[0.8])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.003],
    #                                           specific_pruning_rates=[0.5])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.001],
    #                                           specific_pruning_rates=[0.5])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.001],
    #                                           specific_pruning_rates=[0.8])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.001],
    #                                           specific_pruning_rates=[0.9])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.005],
    #                                           specific_pruning_rates=[0.5])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.005],
    #                                           specific_pruning_rates=[0.8])
    # plot_specific_pr_sigma_epsilon_statistics(fp, cfg, specific_sigmas=[0.005],
    #                                           specific_pruning_rates=[0.9])

    # statistics_of_epsilon_for_stochastic_pruning(fp, cfg)

    # stochastic_pruning_with_sigma_optimization_with_erk_layer_wise_prunig_rates(cfg)
    # stochastic_pruning_with_sigma_and_pr_optimization(cfg)
    # stochastic_pruning_with_sigma_optimization(cfg)
    # transfer_mask_rank_experiments_plot_adaptive_noise(cfg)

    # transfer_mask_rank_experiments(cfg,eval_set="val")
    # weights_analysis_per_weight(cfg)
    # pruning_rates = [ 0.9, 0.8, 0.5]
    # # thresholds = []
    # for pr in pruning_rates:
    #     cfg.amount = pr
    #     threshold = check_noise_impact_in_weights_by_layer(cfg)

    # thresholds.append(threshold)
    # df = pd.DataFrame({"Pruning Rate": pruning_rates, "Threshold": thresholds})
    # df.to_csv("data/analysis_per_layer/pruning_thesholds_per_layer_traditional_trained.csv", sep=",", index=False)
    # check_sigma_normalization_against_weights(cfg)
    # fp = "data/epsilon_experiments_t_1-33_full.csv"
    # statistics_of_epsilon_for_stochastic_pruning(fp, cfg)
    # sigmas = [0.001,0.002,0.005]
    # for sig in sigmas:
    #     cfg.sigma = sig
    #     check_sigma_normalization_againts_weights(cfg)

    # file_path = population_sweeps_transfer_mask_rank_experiments()
    # fp = "data/epsilon_experiments_all_N_sig_0.001_pr_0.9.csv"
    # df1 = pd.read_csv("data/epsilon_experiments_t_1-33pop_10_sig_0.001_pr_0.8.csv",sep=",",header=0)
    # df1 = df1.append(pd.read_csv("data/epsilon_experiments_t_1-33pop_50_sig_0.001_pr_0.8.csv",sep=",",header=0))
    # df1 = df1.append(pd.read_csv("data/epsilon_experiments_t_1-33pop_100_sig_0.001_pr_0.8.csv",sep=",",header=0))
    # df1.to_csv(fp,sep=",", index=False)
    # single_statistics_of_epsilon_for_stochastic_pruning(fp)
    # statistics_of_epsilon_for_stochastic_pruning("data/epsilon_experiments_t_1-33_full.csv", cfg)

    # pruning_rates = [0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    # for pr in pruning_rates:
    #     cfg.amount = pr
    #     main(cfg)

