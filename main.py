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
from matplotlib.ticker import PercentFormatter
import sklearn as sk
from sklearn.manifold import MDS, TSNE
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time


# function taken from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# #sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
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
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(int(data[i, j]), None), **kw)
            texts.append(text)

    return texts


def load_model(net, path):
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
            if name == "":
                prune.remove(m, "weight")
            if name == name_module:
                prune.remove(m, "weight")
                break


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


################################# Noise calibration with optuna @##################################
def calibrate(trial: optuna.trial.Trial) -> float:
    # in theory cfg is available everywhere because it is define on the if name ==__main__ section
    net = None
    if cfg.architecture == "resnet18":
        net = ResNet18()

    load_model(net, "trained_models/cifar10/cifar_csghmc_5.pt")
    sigma_add = trial.suggest_float("sigma_add", 0.0001, 0.1)
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
    average_loss = 0
    for i in range(100):
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
        # I use the sum because I don't care where (in the vector) each noise changes specifically
        loss = ((different_add_noise - different_mul_noise) ** 2).item()
        average_loss = average_loss + (loss - average_loss) / (i + 1)
    return average_loss


def noise_calibration(cfg: omegaconf.DictConfig):
    # distributions = {
    #     "sigma_add": optuna.distributions.FloatDistribution(0.0001, 0.1, log=True),
    #     "sigma_mul": optuna.distributions.FloatDistribution(0.1, 1, log=True),
    # }
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner()
    )

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(calibrate, n_trials=1500)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("\n Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig1 = optuna.visualization.plot_optimization_history(study)
    fig2 = optuna.plot_intermediate_values(study)
    fig3 = optuna.plot_param_importances(study)
    fig4 = optuna.contour_plot(study, params=["sigma_add", "sigma_mul"])

    fig1.savefig("data/figures/opt_history.png")
    fig2.savefig("data/figures/intermediate_values.png")
    fig3.savefig("data/figures/para_importances.png")
    fig4.savefig("data/figures/contour_plot.png")


################################# Layer importance experiments ######################################
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

    print("\n######################### EXPERIMENT B ##################################\n")
    if type_exp == "a":
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
        # matrix = np.random.randn(len(layers),len(prunings_percentages))

        ax = plt.subplot()
        im = ax.imshow(matrix, aspect=0.5)
        # plt.imshow(matrix)
        # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.yticks(ticks=range(0, len(layer_names)), labels=layer_names)
        plt.xticks(ticks=range(0, len(prunings_percentages)), labels=prunings_percentages)
        # ax.set_xticklabels(labels=layer_names,rotation=-45)
        # ax.set_yticklabels(labels=prunings_percentages)
        plt.colorbar(im)
        plt.gcf().set_size_inches(5, 5)
        texts = annotate_heatmap(im, valfmt="{x}")
        plt.tight_layout()
        result = time.localtime(time.time())
        plt.savefig(f"data/figures/layer_V_prune_{result.tm_hour}-{result.tm_min}.pdf")
        plt.close()
        percent_index = matrix.argmax(axis=1)
        best_percentage_for_layers = {}
        for i, name in enumerate(layer_names):
            best_percentage_for_layers[name] = percent_index[i]
        print(f"Best percentage by layer {best_percentage_for_layers}")
        with open(f"data/best_per_layer_{cfg.architecture}.pkl", "wb") as f:
            pickle.dump(best_percentage_for_layers, f)

    if type_exp == "b":
        # Organise the layers to accumulate with the number of weights
        # with open(f"data/best_per_layer_{cfg.architecture}.pkl", "rb") as f:
        #     prune_rate_per_layer = pickle.load(f)
        #
        print("\n######################### EXPERIMENT B##################################\n")
        count = lambda w: w.nelement()
        number_of_elements = list(map(count, weights))
        sorted_by_n = np.argsort(number_of_elements)
        sorted_layer_names = [layers[g] for g in sorted_by_n]
        # sorted_layer_names
        # Then im going to prune all the layers up to layer i with the pruning rate obtained from the type a experiment.
        for i, layer_tuple in enumerate(sorted_layer_names):
            for j, pruning_percentage in enumerate(prunings_percentages):
                current_model = copy.deepcopy(model)

                sub_layers = sorted_layer_names[:i + 1]

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

        ax = plt.subplot()
        im = ax.imshow(matrix, aspect=0.5)
        # plt.imshow(matrix)
        # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        layer_names_by_n, _ = zip(*sorted_layer_names)
        plt.yticks(ticks=range(0, len(layer_names_by_n)), labels=layer_names_by_n)
        plt.xticks(ticks=range(0, len(prunings_percentages)), labels=prunings_percentages)
        # ax.set_xticklabels(labels=layer_names,rotation=-45)
        # ax.set_yticklabels(labels=prunings_percentages)
        plt.colorbar(im)
        plt.gcf().set_size_inches(5, 5)
        texts = annotate_heatmap(im, valfmt="{x}")
        plt.tight_layout()
        result = time.localtime(time.time())
        plt.savefig(f"data/figures/cumsum_layer_V_prune_{result.tm_hour}-{result.tm_min}.pdf")
        plt.close()

        # Now I'm going to do something similar to the above but just prune with the optimal rate


def run_layer_experiment(cfg):
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
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)
    load_model(net, "trained_models/cifar10/cifar_csghmc_5.pt")

    layer_importance_experiments(cfg, net, use_cuda, testloader, type_exp="a")
    layer_importance_experiments(cfg, net, use_cuda, testloader, type_exp="b")


############################################
def main(cfg: omegaconf.DictConfig):
    print("torch version: {}".format(torch.__version__))

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
    original_performance = test(net, use_cuda, testloader)
    pruned_original = copy.deepcopy(net)
    weights_to_prune = weigths_to_prune(pruned_original)
    prune.global_unstructured(
        weights_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=cfg.amount

    )
    remove_reparametrization(pruned_original)
    print("performance of pruned original")
    pruned_original_performance = test(pruned_original, use_cuda, testloader)
    pop.append(pruned_original)
    performance.append(pruned_original_performance)
    # pp = pprint.PrettyPrinter(ident=4)
    for n in range(N):
        current_model = copy.deepcopy(net)
        if cfg.noise == "gaussian":
            current_model.apply(add_gaussian_noise_to_weights)
        elif cfg.noise == "geogaussian":
            current_model.apply(add_geometric_gaussian_noise_to_weights)

        prune.global_unstructured(
            weigths_to_prune(current_model),
            pruning_method=prune.L1Unstructured,
            amount=cfg.amount)
        remove_reparametrization(current_model)
        print("Performance for model {}".format(n))
        current_model.cpu()
        pop.append(current_model)
        performance.append(test(current_model, use_cuda, testloader))
    proba_function = get_proba_function(cfg.amount, count_parameters(net))
    ranked_index = np.flip(np.argsort(performance))
    performance = np.array(performance)
    np.save("data/population_data/performances_{}.npy".format(cfg.noise), performance)
    with open("data/population_data/population_models_{}.pkl".format(cfg.noise), "wb") as f:
        pickle.dump(pop, f)

    cutoff = original_performance - 2

    plt.figure()
    plt.axhline(y=original_performance, color="k", linestyle="-", label="Dense performance")
    plt.axhline(y=cutoff, color="r", linestyle="--", label="cut of value")
    plt.xlabel("Ranking index", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.title("CIFAR10", fontsize=20)
    first = 0
    for i, element in enumerate(performance[ranked_index]):
        if element == pruned_original_performance and first == 0:
            plt.scatter(i, element, c="g", marker="o", label="original model pruned")
            first = 1
        else:
            plt.scatter(i, element, c="b", marker="x")
    plt.legend()
    plt.show()


def plot(cfg):
    performance = np.load("data/performances_{}.npy".format(cfg.noise))
    ranked_index = np.flip(np.argsort(performance))
    cutoff = 92
    plt.figure()
    plt.axhline(y=94.87, color="k", linestyle="-", label="Dense performance")
    plt.axhline(y=cutoff, color="r", linestyle="--", label="cut-off value")
    plt.xlabel("Ranking index", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    maximum = np.max(performance)
    for i, element in enumerate(performance[ranked_index]):
        if element == maximum:
            plt.scatter(i, element, c="g", marker="o", label="original model pruned")
        else:
            plt.scatter(i, element, c="b", marker="x")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    cfg = omegaconf.DictConfig({
        "population": 10,
        "architecture": "resnet18",
        "noise": "geogaussian",
        "amount": 0.5,
        "use_wandb": True
    })
    noise_calibration(cfg)
    # run_layer_experiment(cfg)
    # weight_inspection(cfg, 90)
    # save_onnx(cfg)
