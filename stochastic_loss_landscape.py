from main import get_datasets, get_model, prune_with_rate, get_noisy_sample, remove_reparametrization, test, \
    get_random_batch, sparsity, get_layer_dict
import argparse
from shrinkbench.metrics.flops import flops
import wandb
import omegaconf
from omegaconf import DictConfig
import copy
import torch
import os
from itertools import cycle
import loss_landscapes
import loss_landscapes.metrics as metrics
import numpy as np
import time
import re
import torchessian as torchessian
import pandas as pd
import torch
import numpy as np
import pickle
import matplotlib.ticker as ticker
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sparse_ensemble_utils import test

steps = 200
low_t = -7
high_t = 7
# b = 1.5
fs = 12
fig_size = (5, 3)
sns.reset_orig()
sns.reset_defaults()
matplotlib.rc_file_defaults()
plt.rcParams.update({
    "axes.linewidth": 0.5,
    'axes.edgecolor': 'black',
    "grid.linewidth": 0.4,
    "lines.linewidth": 1,
    'xtick.bottom': True,
    'xtick.color': 'black',
    "xtick.direction": "out",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    'ytick.left': True,
    'ytick.color': 'black',
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "figure.figsize": [3.3, 2.5],
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{bm} \usepackage{amsmath}",
})
plt.grid(ls='--', alpha=0.5)
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


def load_cfg(args):
    solution = ""
    exclude_layers = None
    if args["dataset"] == "cifar100":
        if args["modeltype"] == "alternative":
            if args["architecture"] == "resnet18":
                solution = "trained_models/cifar100/resnet18_cifar100_traditional_train.pth"
                exclude_layers = ["conv1", "linear"]
            if args["architecture"] == "VGG19":
                solution = "trained_models/cifar100/vgg19_cifar100_traditional_train.pth"
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = "trained_models/cifar100/resnet50_cifar100.pth"
                exclude_layers = ["conv1", "linear"]
    if args["dataset"] == "cifar10":
        if args["modeltype"] == "alternative":
            if args["architecture"] == "resnet18":
                solution = "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth"
                # solution = "trained_models/cifar10/resnet18_cifar10_normal_seed_3.pth"
                exclude_layers = ["conv1", "linear"]
            if args["architecture"] == "VGG19":
                solution = "trained_models/cifar10/VGG19_cifar10_traditional_train_valacc=93,57.pth"
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = "trained_models/cifar10/resnet50_cifar10.pth"
                exclude_layers = ["conv1", "linear"]
        if args["modeltype"] == "hub":
            if args["architecture"] == "resnet18":
                solution = "trained_models/cifar10/resnet18_official_cifar10_seed_2_test_acc_88.51.pth"
                exclude_layers = ["conv1", "fc"]

    if args["dataset"] == "imagenet":

        if args["modeltype"] == "hub":

            if args["architecture"] == "resnet18":
                solution = "/nobackup/sclaam/trained_models/resnet18_imagenet.pth"
                exclude_layers = ["conv1", "fc"]
                # exclude_layers = []
            if args["architecture"] == "VGG19":
                raise NotImplementedError("Not implemented")
                solution = "trained_models/cifar100/vgg19_cifar100_traditional_train.pth"
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = "/nobackup/sclaam/trained_models/resnet50_imagenet.pth"
                exclude_layers = ["conv1", "fc"]

    cfg = omegaconf.DictConfig({
        "population": args["population"],
        "generations": 10,
        "epochs": args["epochs"],
        "short_epochs": 10,
        "architecture": args["architecture"],
        "solution": solution,
        "noise": "gaussian",
        "pruner": args["pruner"],
        "model_type": args["modeltype"],
        "fine_tune_exclude_layers": True,
        "fine_tune_non_zero_weights": True,
        "sampler": "tpe",
        "flop_limit": 0,
        "one_batch": True,
        "measure_gradient_flow": True,
        "full_fine_tune": False,
        "use_stochastic": True,
        "sigma": args["sigma"],
        "noise_after_pruning": 0,
        "amount": args["pruning_rate"],
        "dataset": args["dataset"],
        "batch_size": args["batch_size"],
        "num_workers": args["num_workers"],
        "save_model_path": "stochastic_pruning_models/",
        "save_data_path": "stochastic_pruning_data/",
        "gradient_cliping": True,
        "use_wandb": False,
        "id": args["id"],
    })

    cfg.exclude_layers = exclude_layers
    return cfg


def W_hat(begining, end, w):
    return begining + w * (end - begining)


def little_test(outputs, targets):
    total = 0
    correct = 0
    _, predicted = torch.max(outputs, 1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    # print(correct)
    # print(total)
    return torch.tensor(100 - 100 * (correct / total))


def linear_interpolation_oneshot_GMP(cfg, eval_set="val", print_exclude_layers=True):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    exclude_layers_string = "_exclude_layers" if print_exclude_layers else ""
    # if cfg.use_wandb:
    #     os.environ["wandb_start_method"] = "thread"
    #     # now = date.datetime.now().strftime("%m:%s")
    #     wandb.init(
    #         entity="luis_alfredo",
    #         config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
    #         project="stochastic_pruning",
    #         name=f"one_shot_stochastic_pruning_static_sigma_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}",
    #         notes="This experiment is to test if iterative global stochastic pruning, compares to one-shot stochastic pruning",
    #         reinit=True,
    #     )

    pruned_model = get_model(cfg)
    pruned_model.cuda()
    det_pruning_model = copy.deepcopy(pruned_model)
    names, weights = zip(*get_layer_dict(det_pruning_model))
    number_of_layers = len(names)

    if cfg.pruner == "global":
        prune_with_rate(det_pruning_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(det_pruning_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    remove_reparametrization(det_pruning_model, exclude_layer_list=cfg.exclude_layers)
    # record_predictions(pruned_original, evaluation_set,
    #                    "{}_one_shot_det_{}_predictions_{}".format(cfg.architecture, cfg.model_type, cfg.dataset))
    print("pruned_performance of pruned original")
    deterministic_pruned_performance = test(det_pruning_model, use_cuda, testloader, verbose=1)

    best_model = None
    best_accuracy = 0
    initial_flops = 0
    data_loader_iterator = cycle(iter(valloader))
    data, y = next(data_loader_iterator)
    first_iter = 1
    unit_sparse_flops = 0
    # evaluation_set = None
    # if cfg.one_batch:
    #     evaluation_set = [data]
    # else:
    #     if eval_set == "test":
    #         evaluation_set = testloader
    #     if eval_set == "val":
    #         evaluation_set = valloader

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
        # if first_iter:
        #     _, unit_sparse_flops = flops(noisy_sample, data.cuda())
        #     first_iter = 0

        noisy_sample_performance = test(noisy_sample, use_cuda, valloader, verbose=0,
                                        count_flops=False, batch_flops=unit_sparse_flops)
        print(noisy_sample_performance)
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
    #####################################################################
    #               Now the linear interpolation and the calculation of the loss
    #####################################################################
    det_pruning_model.cuda()
    best_model.cuda()
    vector_deterministic_pruning = torch.nn.utils.parameters_to_vector(det_pruning_model.parameters())
    vector_stochastic_pruning = torch.nn.utils.parameters_to_vector(best_model.parameters())

    begining = torch.lerp(vector_stochastic_pruning, vector_deterministic_pruning, low_t).to("cuda")
    end = torch.lerp(vector_stochastic_pruning, vector_deterministic_pruning, high_t).to("cuda")
    model_begining = copy.deepcopy(pruned_model)
    torch.nn.utils.vector_to_parameters(begining, model_begining.parameters())
    model_end = copy.deepcopy(pruned_model)
    torch.nn.utils.vector_to_parameters(end, model_end.parameters())
    stochastic_loss_index = int((-low_t * steps) // (high_t - low_t))
    criterion = torch.nn.CrossEntropyLoss()
    metric_trainloader = metrics.sl_metrics.BatchedLoss(criterion, trainloader)
    metric_testloader = metrics.sl_metrics.BatchedAccuracyError(test, testloader)
    print("Calculating train loss")
    print("Done!")
    loss_data_train = loss_landscapes.linear_interpolation(model_begining, model_end, metric_trainloader, steps)
    print("Calculating test loss")
    loss_data_test = loss_landscapes.linear_interpolation(model_begining, model_end, metric_testloader, steps)
    print(loss_data_test)
    print("Done!")
    identifier = cfg.id
    name = "{}_{}_{}_{}_{}".format(identifier, cfg.dataset, cfg.architecture, cfg.sigma, cfg.amount)
    np.save("/nobackup/sclaam/smoothness/trainloss_line_{}_one_shot.npy".format(name), loss_data_train)
    np.save("/nobackup/sclaam/smoothness/testAccuracy_line_{}_one_shot.npy".format(name), loss_data_test)
    torch.save(best_model.state_dict(), "/nobackup/sclaam/smoothness/{}_stochastic_one_shot.pth".format(name))
    torch.save(det_pruning_model.state_dict(), "/nobackup/sclaam/smoothness/{}_deterministic_one_shot.pth".format(name))


def linear_interpolation_dense_GMP(cfg, eval_set="test", print_exclude_layers=True):
    trainloader, valloader, testloader = get_datasets(cfg)
    target_sparsity = cfg.amount
    use_cuda = torch.cuda.is_available()
    exclude_layers_string = "_exclude_layers" if print_exclude_layers else ""
    # if cfg.use_wandb:
    #     os.environ["wandb_start_method"] = "thread"
    #     # now = date.datetime.now().strftime("%m:%s")
    #     wandb.init(
    #         entity="luis_alfredo",
    #         config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
    #         project="stochastic_pruning",
    #         name=f"one_shot_stochastic_pruning_static_sigma_{cfg.pruner}_pr_{cfg.amount}{exclude_layers_string}",
    #         notes="This experiment is to test if iterative global stochastic pruning, compares to one-shot stochastic pruning",
    #         reinit=True,
    #     )

    pruned_model = get_model(cfg)
    pruned_model.cuda()
    det_pruning_model = copy.deepcopy(pruned_model)
    names, weights = zip(*get_layer_dict(det_pruning_model))
    number_of_layers = len(names)

    # if cfg.pruner == "global":
    #     prune_with_rate(det_pruning_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")
    # else:
    #     prune_with_rate(det_pruning_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
    #                     pruner=cfg.pruner)
    #
    # remove_reparametrization(det_pruning_model, exclude_layer_list=cfg.exclude_layers)
    # record_predictions(pruned_original, evaluation_set,
    #                    "{}_one_shot_det_{}_predictions_{}".format(cfg.architecture, cfg.model_type, cfg.dataset))
    print("pruned_performance of pruned original")
    deterministic_pruned_performance = test(det_pruning_model, use_cuda, testloader, verbose=1)

    best_model = None
    best_accuracy = 0
    initial_flops = 0
    data_loader_iterator = cycle(iter(valloader))
    data, y = next(data_loader_iterator)
    first_iter = 1
    unit_sparse_flops = 0
    # evaluation_set = None
    # if cfg.one_batch:
    #     evaluation_set = [data]
    # else:
    #     if eval_set == "test":
    #         evaluation_set = testloader
    #     if eval_set == "val":
    #         evaluation_set = valloader

    for n in range(cfg.population):

        noisy_sample = get_noisy_sample(pruned_model, cfg)

        # det_mask_transfer_model = copy.deepcopy(current_model)
        # copy_buffers(from_net=pruned_original, to_net=det_mask_transfer_model)
        # det_mask_transfer_model_performance = test(det_mask_transfer_model, use_cuda, evaluation_set, verbose=1)

        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        # Dense stochastic performance
        # if cfg.pruner == "global":
        #     prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
        # else:
        #     prune_with_rate(noisy_sample, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
        #                     pruner=cfg.pruner)
        # # Here is where I transfer the mask from the pruned stochastic model to the
        # # original weights and put it in the ranking
        # # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        # remove_reparametrization(noisy_sample, exclude_layer_list=cfg.exclude_layers)
        # if first_iter:
        #     _, unit_sparse_flops = flops(noisy_sample, data)
        #     first_iter = 0

        noisy_sample_performance = test(noisy_sample, use_cuda, valloader, verbose=0,
                                        count_flops=False, batch_flops=unit_sparse_flops)

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
    #####################################################################
    #               Now the linear interpolation and the calculation of the loss
    #####################################################################
    det_pruning_model.cuda()
    best_model.cuda()
    vector_deterministic_pruning = torch.nn.utils.parameters_to_vector(det_pruning_model.parameters())
    vector_stochastic_pruning = torch.nn.utils.parameters_to_vector(best_model.parameters())

    begining = torch.lerp(vector_stochastic_pruning, vector_deterministic_pruning, low_t).to("cuda")
    end = torch.lerp(vector_stochastic_pruning, vector_deterministic_pruning, high_t).to("cuda")
    model_begining = copy.deepcopy(pruned_model)
    torch.nn.utils.vector_to_parameters(begining, model_begining.parameters())
    model_end = copy.deepcopy(pruned_model)
    torch.nn.utils.vector_to_parameters(end, model_end.parameters())
    stochastic_loss_index = int((-low_t * steps) // (high_t - low_t))
    criterion = torch.nn.CrossEntropyLoss()
    metric_trainloader = metrics.sl_metrics.BatchedLoss(criterion, trainloader)
    metric_testloader = metrics.sl_metrics.BatchedAccuracyError(test, testloader)
    print("Calculating train loss")
    loss_data_train = loss_landscapes.linear_interpolation(model_begining, model_end, metric_trainloader, steps)
    print("Done!")
    print("Calculating test loss")
    loss_data_test = loss_landscapes.linear_interpolation(model_begining, model_end, metric_testloader, steps)
    print("Done!")
    identifier = cfg.id
    name = "{}_{}_{}_{}_{}".format(identifier, cfg.dataset, cfg.architecture, cfg.sigma, cfg.amount)
    np.save("/nobackup/sclaam/smoothness/trainloss_line_{}_dense.npy".format(name), loss_data_train)
    np.save("/nobackup/sclaam/smoothness/testAccuracy_line_{}_dense.npy".format(name), loss_data_test)
    torch.save(best_model.state_dict(), "/nobackup/sclaam/smoothness/{}_stochastic_dense.pth".format(name))
    torch.save(det_pruning_model.state_dict(), "/nobackup/sclaam/smoothness/{}_deterministic_dense.pth".format(name))


def plot_line_(cfg, type_="one_shot"):
    model = get_model(cfg)
    identifier = cfg.id
    name = "{}_{}_{}_{}_{}".format(identifier, cfg.dataset, cfg.architecture, cfg.sigma, cfg.amount)
    train_loss = np.load("smoothness/trainloss_line_{}_{}.npy".format(name, type_))
    test_loss = np.load("smoothness/testAccuracy_line_{}_{}.npy".format(name, type_))
    stochastic_state_dict = torch.load("smoothness/{}_stochastic_{}.pth".format(name, type_))
    deterministic_state_dict = torch.load("smoothness/{}_deterministic_{}.pth".format(name, type_))
    sto_model = copy.deepcopy(model)
    det_model = copy.deepcopy(model)
    sto_model.load_state_dict(stochastic_state_dict)
    det_model.load_state_dict(deterministic_state_dict)
    vector_deterministic_pruning = torch.nn.utils.parameters_to_vector(det_model.parameters())
    vector_stochastic_pruning = torch.nn.utils.parameters_to_vector(sto_model.parameters())
    # t_range = torch.arange(low_t, high_t, (high_t - low_t) / (steps))
    stochastic_loss_index = int((-low_t * steps) // (high_t - low_t))
    # deterministic_loss_index = int((1.5 * steps) // 2) if low
    deterministic_loss_index = int(((1 - low_t) * steps) // (high_t - low_t))
    #
    # begining = torch.lerp(vector_stochastic_pruning, vector_deterministic_pruning, t_range[stochastic_loss_index]).to(
    #     "cuda")
    # end = torch.lerp(vector_stochastic_pruning, vector_deterministic_pruning,t_range[deterministic_loss_index]).to("cuda")

    # point = torch.lerp(vector_stochastic_pruning, vector_deterministic_pruning, -0.5)
    t_range = torch.arange(low_t, high_t, (high_t - low_t) / (steps))
    distance_vector = []
    for t in t_range:
        point = torch.lerp(vector_stochastic_pruning, vector_deterministic_pruning, t)
        distance = torch.norm(vector_stochastic_pruning - point)
        if t <= 0:
            distance_vector.append(-distance.detach().cpu().numpy())
        else:
            distance_vector.append(distance.detach().cpu().numpy())

    fig, ax = plt.subplots(figsize=fig_size, layout="compressed")
    plt.plot(distance_vector, train_loss, color="cornflowerblue", label="Train Loss")
    ax2 = plt.twinx()
    ax2.plot(distance_vector, test_loss, color="limegreen", label="Test Accuracy error")
    # plt.legend()
    # plt.show()
    if "one_shot" in type_:
        plt.scatter(distance_vector[stochastic_loss_index], train_loss[stochastic_loss_index], c="cornflowerblue",
                    marker="o", label="Stochastic pruning")
        plt.scatter(distance_vector[deterministic_loss_index], train_loss[deterministic_loss_index], c="cornflowerblue",
                    marker="s", label="Deterministic pruning")
        plt.scatter(distance_vector[stochastic_loss_index], test_loss[stochastic_loss_index], c="limegreen",
                    marker="o", label="Stochastic pruning")
        plt.scatter(distance_vector[deterministic_loss_index], test_loss[deterministic_loss_index], c="limegreen",
                    marker="s", label="Deterministic pruning")
    elif "dense" in type_:
        plt.scatter(distance_vector[stochastic_loss_index], train_loss[stochastic_loss_index], c="cornflowerblue",
                    marker="o", label="Stochastic")
        plt.scatter(distance_vector[deterministic_loss_index], train_loss[deterministic_loss_index], c="cornflowerblue",
                    marker="s", label="Deterministic")
        ax2.scatter(distance_vector[stochastic_loss_index], test_loss[stochastic_loss_index], c="limegreen",
                    marker="o", label="Stochastic")
        ax2.scatter(distance_vector[deterministic_loss_index], test_loss[deterministic_loss_index], c="limegreen",
                    marker="s", label="Deterministic")
    plt.legend()
    plt.savefig("paper_plots/line_{}_{}.pdf".format(name, type_))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stochastic pruning experiments')
    parser.add_argument('-pop', '--population', type=int, default=1, help='Population', required=False)
    parser.add_argument('-gen', '--generation', type=int, default=10, help='Generations', required=False)
    # parser.add_argument('-mod', '--model_type',type=str,default=alternative, help = 'Type of model to use', required=False)
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Epochs for fine tuning', required=False)
    parser.add_argument('-sig', '--sigma', type=float, default=0.005, help='Noise amplitude', required=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='Batch size', required=True)
    parser.add_argument('-pr', '--pruner', type=str, default="global", help='Type of prune', required=True)
    parser.add_argument('-dt', '--dataset', type=str, default="cifar10", help='Dataset for experiments', required=True)
    parser.add_argument('-ar', '--architecture', type=str, default="resnet18", help='Type of architecture',
                        required=True)
    # parser.add_argument('-so', '--solution',type=str,default="", help='Path to the pretrained solution, it must be consistent with all the other parameters', required=True)
    parser.add_argument('-mt', '--modeltype', type=str, default="alternative",
                        help='The type of model (which model definition/declaration) to use in the architecture',
                        required=True)
    parser.add_argument('-pru', '--pruning_rate', type=float, default=0.9, help='percentage of weights to prune',
                        required=False)
    ############# this is for pr and sigma optim ###############################
    parser.add_argument('-nw', '--num_workers', type=int, default=4, help='Number of workers', required=False)
    parser.add_argument('-ob', '--one_batch', type=bool, default=False, help='One batch in sigma pr optim',
                        required=False)

    parser.add_argument('-sa', '--sampler', type=str, default="tpe", help='Sampler for pr sigma optim', required=False)
    parser.add_argument('-ls', '--log_sigma', type=bool, default=False,
                        help='Use log scale for sigma in pr,sigma optim', required=False)
    parser.add_argument('-tr', '--trials', type=int, default=300, help='Number of trials for sigma,pr optim',
                        required=False)
    parser.add_argument('-fnc', '--functions', type=int, default=1,
                        help='Type of functions for MOO optim of sigma and pr', required=False)
    parser.add_argument('-id', '--id', type=str, default="", help='Unique identifier', required=True)
    parser.add_argument('-tp', '--type', type=str, default="one_shot", help='Type', required=False)

    args = vars(parser.parse_args())
    cfg = load_cfg(args)
    linear_interpolation_oneshot_GMP(cfg)
    linear_interpolation_dense_GMP(cfg)
    # plot_line_(cfg, args["type"])
