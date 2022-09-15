import sys
import pickle

sys.path.append('csgmcmc')
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
from models import *
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.utils.prune as prune
import platform
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.manifold import MDS,TSNE


def load_model(net, path):
    net.load_state_dict(torch.load(path))

def get_layer_dict(model):
    iter_1 = model1.named_modules()
    layer_dict = []


    for name, m in iter_1:
        with torch.no_grad():
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m,nn.BatchNorm2d) and not \
                    isinstance(m, nn.BatchNorm3d):
                    layer_dict.append((name,torch.flatten(m).cpu().detach()))

    return layer_dict



def models_inspection(population):

    layers_of_models = [get_layer_dict(mod) for mod in population]
    sparsity = lambda w: np.sum(w == -1) / len(w)
    tsne = []
    mds = []
    MDS_embeddings = MDS(n_components=2)
    tsne_embeddings = TSNE(n_components=2)
    for i,layer in enumerate(layers_of_models):
        names,weigths = list(*layer)
        sparsities = list(map(sparsity,weigths))
        plt.bar(x = sparsities,tick_label = names)
        plt.savefig("data/figures/bar_plot_model_{}.png".format(i))
        plt.close()

        fig, ax = plt.subplots()
        for element in layer:
            #MDS.append("Model {} {}".format(i,element[0]),MDS_embeddings.fit_transform(element[1]))
            embedding = tsne_embeddings.fit_transform(element[1])
            ax.scatter(embedding)
            label ="M:{}L:{}".format(i,element[0])
            ax.annotate(label, (embedding[0], embedding[1]))
        plt.savefig("data/figures/layer_projection_tsne.png")
        plt.close()







def weight_inspection(cfg, cutoff):
    """
    This function inspects statistics about the weights of the pruned models
    :return:
    """
    performances = np.load("data/performances_{}.npy".format(cfg.noise))
    with open("data/population_models_{}.pkl".format(cfg.noise), "rb") as f:
        pop = pickle.load(f)
    sorted_indx = np.argsort(performances)
    sorted_models = [pop[i] for i in sorted_indx]

    above_cutoff_index= np.where(performances[sorted_indx] >= cutoff)[0]
    underneath_cutoff_index = np.where(performances[sorted_indx] < cutoff)[0]
    measure_overlap(sorted_models[0],sorted_models[1])
    models_inspection(sorted_models[:5])

    # Go trough all the above_cutoff_index



def add_geometric_gaussian_noise_to_weights(m):
    with torch.no_grad():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            m.weight.multiply_(torch.normal(mean=torch.ones_like(m.weight), std=0.2).to(m.weight.device))


def add_gaussian_noise_to_weights(m):
    with torch.no_grad():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            m.weight.add_(torch.normal(mean=torch.zeros_like(m.weight), std=0.01).to(m.weight.device))


def test(net, use_cuda, testloader):
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


def remove_reparametrization(model):
    for m in model.modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d):
            prune.remove(m, "weight")


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


def main(cfg: omegaconf.DictConfig):
    print("torch version: {}".format(torch.__version__))

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
    np.save("data/performances_{}.npy".format(cfg.noise), performance)
    with open("data/population_models_{}.pkl".format(cfg.noise), "wb") as f:
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
        "noise": "geogaussian",
        "amount": 0.5
    })

    weight_inspection(cfg,90)
