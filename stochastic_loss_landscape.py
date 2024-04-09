from main import get_datasets, get_model, prune_with_rate, get_noisy_sample, remove_reparametrization, test, \
    get_random_batch, sparsity, get_layer_dict, prune_function
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
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.ticker as ticker
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import matplotlib.lines as mlines
from sparse_ensemble_utils import test
import math

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


def test_linear_interpolation(model, beg_vector, end_vector, t_range, test_function, test_loder, cfg):
    output = []
    model_copy = copy.deepcopy(model)
    for t in t_range:
        point = torch.lerp(beg_vector, end_vector, t.cuda()).to("cuda")
        torch.nn.utils.vector_to_parameters(point, model_copy.parameters())
        accuracy = test_function(model_copy, True, test_loder, verbose=0)
        print(100 - accuracy)
        output.append(100 - accuracy)
    return np.array(output)


def loss_cross_entropy(net, use_cuda, testloader, one_batch=False, verbose=2, count_flops=False, batch_flops=0,
                       number_batches=0):
    if use_cuda:
        net.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    if count_flops:
        assert batch_flops != 0, "If count_flops is True,batch_flops must be non-zero"

    sparse_flops = 0
    first_time = 1
    sparse_flops_batch = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if count_flops:
                sparse_flops += batch_flops
            test_loss += loss.data.item()
            if torch.all(outputs > 0):
                _, predicted = torch.max(outputs.data, 1)
            else:
                soft_max_outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(soft_max_outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            # print(correct/total)

            if batch_idx % 100 == 0:
                if verbose == 2:
                    print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                          % (test_loss / (batch_idx + 1), 100. * correct.item() / total, correct, total))
            if one_batch:
                if count_flops:
                    return 100. * correct.item() / total, sparse_flops
                else:
                    return 100. * correct.item() / total

            if number_batches > 0:
                if number_batches < batch_idx:
                    return 100. * correct.item() / total

    if verbose == 1 or verbose == 2:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss / len(testloader), correct, total,
            100. * correct.item() / total))
    # net.cpu()
    if count_flops:
        return test_loss / len(testloader), sparse_flops
    else:
        return test_loss / len(testloader)


def test_linear_interpolation_pruned_models(model, beg_vector, end_vector, t_range, test_function, test_loder):
    output_dense = []
    output_sparse = []
    model_copy = copy.deepcopy(model)

    for t in t_range:
        point = torch.lerp(beg_vector, end_vector, t.cuda()).to("cuda")
        torch.nn.utils.vector_to_parameters(point, model_copy.parameters())
        dense_accuracy = test_function(model_copy, True, test_loder, verbose=0)
        prune_function(model_copy, cfg)
        remove_reparametrization(model_copy, exclude_layer_list=cfg.exclude_layers)
        sparse_accuracy = test_function(model_copy, True, test_loder, verbose=0)

        print("Dense error :{}".format(100 - dense_accuracy))

        print("Sparse error :{}".format(100 - sparse_accuracy))

        output_dense.append(100 - dense_accuracy)
        output_sparse.append(100 - sparse_accuracy)

    return np.array(output_dense), np.array(output_sparse)


def train_linear_interpolation_pruned_models(model, beg_vector, end_vector, t_range, loss_function, test_loder):
    output_dense = []
    output_sparse = []
    model_copy = copy.deepcopy(model)

    for t in t_range:
        point = torch.lerp(beg_vector, end_vector, t.cuda()).to("cuda")
        torch.nn.utils.vector_to_parameters(point, model_copy.parameters())
        dense_accuracy = loss_function(model_copy, True, test_loder, verbose=0)
        prune_function(model_copy, cfg)
        remove_reparametrization(model_copy, exclude_layer_list=cfg.exclude_layers)
        sparse_accuracy = loss_function(model_copy, True, test_loder, verbose=0)

        print("Dense error :{}".format(100 - dense_accuracy))

        print("Sparse error :{}".format(100 - sparse_accuracy))

        output_dense.append(100 - dense_accuracy)
        output_sparse.append(100 - sparse_accuracy)

    return np.array(output_dense), np.array(output_sparse)


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
    dense_model_performance = test(pruned_model, use_cuda, testloader, verbose=0)
    print("Dense model performance{}".format(dense_model_performance))
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
    deterministic_pruned_performance = test(det_pruning_model, use_cuda, testloader, verbose=1)
    print("pruned_performance of pruned original (deterministic) {}".format(deterministic_pruned_performance))
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

        noisy_sample_performance = test(noisy_sample, use_cuda, valloader, verbose=1,
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
    test_accuracy = test(best_model, use_cuda, testloader, verbose=0)
    print("Stochastic performance: {}".format(test_accuracy))
    vector_stochastic_pruning = torch.nn.utils.parameters_to_vector(best_model.parameters()).detach()
    vector_deterministic_pruning = torch.nn.utils.parameters_to_vector(det_pruning_model.parameters()).detach()

    begining = torch.lerp(vector_stochastic_pruning, vector_deterministic_pruning, low_t).to("cuda").detach()
    end = torch.lerp(vector_stochastic_pruning, vector_deterministic_pruning, high_t).to("cuda").detach()
    print("Distance between end and determinsitic (end): {}".format(torch.norm(end - vector_deterministic_pruning)))
    print("Distance between beginning and stochastic (beginning): {}".format(
        torch.norm(begining - vector_stochastic_pruning)))
    model_begining = copy.deepcopy(best_model)
    torch.nn.utils.vector_to_parameters(begining, model_begining.parameters())
    test_vect = torch.nn.utils.parameters_to_vector(model_begining.parameters())

    print("Distance between test vect and stochastic (beginning): {}".format(
        torch.norm(test_vect - vector_stochastic_pruning)))
    model_end = copy.deepcopy(det_pruning_model)
    torch.nn.utils.vector_to_parameters(end, model_end.parameters())
    stochastic_loss_index = math.floor((-low_t * steps) / (high_t - low_t))
    # deterministc_loss_index = int((-low_t * steps) // (high_t - low_t))
    deterministic_loss_index = math.floor(((1 - low_t) * steps) // (high_t - low_t))

    begining_accuracy = test(model_begining, use_cuda, testloader, verbose=0)
    print("Beginning Accuracy: {}".format(begining_accuracy))
    end_accuracy = test(model_end, use_cuda, testloader, verbose=0)
    print("End Accuracy: {}".format(end_accuracy))
    criterion = torch.nn.CrossEntropyLoss()

    metric_trainloader = metrics.sl_metrics.BatchedLoss(criterion, trainloader)
    metric_testloader = metrics.sl_metrics.BatchedAccuracyError(test, testloader)
    print("Calculating train loss")
    print("Done!")
    loss_data_train = loss_landscapes.linear_interpolation(model_begining, model_end, metric_trainloader, steps)
    print(loss_data_train)
    print("Calculating test loss")
    # t0 = time.time()
    # loss_data_test, point_stochastic = loss_landscapes.linear_interpolation(model_begining, model_end,
    #                                                                         metric_testloader, steps,
    #                                                                         reference_model1=stochastic_loss_index - 1)

    # closest_to_stochastic_point_vector = torch.nn.utils.parameters_to_vector(point_stochastic.parameters())
    # print(loss_data_test)
    # print("distance between closest point in the interpolation and the actual stochastic point {}".format(
    #     torch.norm(closest_to_stochastic_point_vector - vector_stochastic_pruning)))
    #
    # loss_data_test, point_deterministic = loss_landscapes.linear_interpolation(model_begining, model_end,
    #                                                                            metric_testloader, steps,
    #                                                                            reference_model1=deterministic_loss_index - 1)
    # closest_to_deterministic_point_vector = torch.nn.utils.parameters_to_vector(point_deterministic.parameters())
    # print("distance between closest point in the interpolation and the actual deterministic point {}".format(
    # torch.norm(closest_to_deterministic_point_vector - vector_deterministic_pruning)))
    # print(loss_data_test
    # t1 = time.time()
    # print("Total time: {}".format(t1 - t0))
    loss_data_test = test_linear_interpolation(det_pruning_model, vector_stochastic_pruning,
                                               vector_deterministic_pruning,
                                               torch.arange(low_t, high_t, (high_t - low_t) / (steps)), test,
                                               testloader)
    print(loss_data_test)
    print("Done!")
    identifier = cfg.id
    name = "{}_{}_{}_{}_{}".format(identifier, cfg.dataset, cfg.architecture, cfg.sigma, cfg.amount)
    np.save("/nobackup/sclaam/smoothness/trainloss_line_{}_one_shot.npy".format(name), loss_data_train)
    np.save("/nobackup/sclaam/smoothness/testAccuracy_line_{}_one_shot.npy".format(name), loss_data_test)
    # np.save("smoothness/testAccuracy_line_{}_one_shot.npy".format(name), loss_data_test)
    torch.save(best_model.state_dict(), "/nobackup/sclaam/smoothness/{}_stochastic_one_shot.pth".format(name))
    # torch.save(best_model.state_dict(), "smoothness/{}_stochastic_one_shot.pth".format(name))
    # torch.save(best_model.state_dict(), "smoothness/{}_stochastic_one_shot.pth".format(name))
    # torch.save(det_pruning_model.state_dict(), "/nobackup/sclaam/smoothness/{}_deterministic_one_shot.pth".format(name))
    torch.save(det_pruning_model.state_dict(), "/nobackup/sclaam/smoothness/{}_deterministic_one_shot.pth".format(name))


def left_pruning_experiments(cfg, eval_set="test", print_exclude_layers=True):
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
    model_begining = copy.deepcopy(best_model)
    torch.nn.utils.vector_to_parameters(begining, model_begining.parameters())
    model_end = copy.deepcopy(det_pruning_model)
    torch.nn.utils.vector_to_parameters(end, model_end.parameters())
    stochastic_loss_index = int((-low_t * steps) // (high_t - low_t))
    criterion = torch.nn.CrossEntropyLoss()
    metric_trainloader = metrics.sl_metrics.BatchedLoss(criterion, trainloader)
    metric_testloader = metrics.sl_metrics.BatchedAccuracyError(test, testloader)
    # print("Calculating train loss")
    # loss_data_train = loss_landscapes.linear_interpolation(model_begining, model_end, metric_trainloader, steps)
    dense_loss_data_train, sparse_loss_data_train = train_linear_interpolation_pruned_models(det_pruning_model,
                                                                                             vector_stochastic_pruning,
                                                                                             vector_deterministic_pruning,
                                                                                             torch.arange(low_t, high_t,
                                                                                                          (
                                                                                                                  high_t - low_t) / (
                                                                                                              steps)),
                                                                                             loss_cross_entropy,
                                                                                             trainloader)
    print("Done!")
    print("Calculating test loss")
    # loss_data_test = loss_landscapes.linear_interpolation(model_begining, model_end, metric_testloader, steps)

    dense_loss_data_test, sparse_loss_data_test = test_linear_interpolation_pruned_models(det_pruning_model,
                                                                                          vector_stochastic_pruning,
                                                                                          vector_deterministic_pruning,
                                                                                          torch.arange(low_t, high_t, (
                                                                                                  high_t - low_t) / (
                                                                                                           steps)),
                                                                                          test,
                                                                                          testloader)
    print("Done!")
    identifier = cfg.id
    name = "{}_{}_{}_{}_{}".format(identifier, cfg.dataset, cfg.architecture, cfg.sigma, cfg.amount)
    np.save("/nobackup/sclaam/smoothness/trainloss_left_line_{}_dense.npy".format(name), dense_loss_data_train)
    np.save("/nobackup/sclaam/smoothness/trainloss_left_line_{}_sparse.npy".format(name), sparse_loss_data_train)
    np.save("/nobackup/sclaam/smoothness/testAccuracy_left_line_{}_dense.npy".format(name), dense_loss_data_test)
    np.save("/nobackup/sclaam/smoothness/testAccuracy_left_line_{}_sparse.npy".format(name), sparse_loss_data_test)
    # np.save("smoothness/testAccuracy_line_{}_dense.npy".format(name), loss_data_test)
    torch.save(best_model.state_dict(), "/nobackup/sclaam/smoothness/{}_left_stochastic_dense.pth".format(name))
    # torch.save(best_model.state_dict(), "smoothness/{}_stochastic_dense.pth".format(name))
    torch.save(det_pruning_model.state_dict(),
               "/nobackup/sclaam/smoothness/{}_left_deterministic_dense.pth".format(name))


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
    model_begining = copy.deepcopy(best_model)
    torch.nn.utils.vector_to_parameters(begining, model_begining.parameters())
    model_end = copy.deepcopy(det_pruning_model)
    torch.nn.utils.vector_to_parameters(end, model_end.parameters())
    stochastic_loss_index = int((-low_t * steps) // (high_t - low_t))
    criterion = torch.nn.CrossEntropyLoss()
    metric_trainloader = metrics.sl_metrics.BatchedLoss(criterion, trainloader)
    metric_testloader = metrics.sl_metrics.BatchedAccuracyError(test, testloader)
    print("Calculating train loss")
    loss_data_train = loss_landscapes.linear_interpolation(model_begining, model_end, metric_trainloader, steps)
    print("Done!")
    print("Calculating test loss")
    # loss_data_test = loss_landscapes.linear_interpolation(model_begining, model_end, metric_testloader, steps)

    loss_data_test = test_linear_interpolation(det_pruning_model, vector_stochastic_pruning,
                                               vector_deterministic_pruning,
                                               torch.arange(low_t, high_t, (high_t - low_t) / (steps)), test,
                                               testloader)
    print("Done!")
    identifier = cfg.id
    name = "{}_{}_{}_{}_{}".format(identifier, cfg.dataset, cfg.architecture, cfg.sigma, cfg.amount)
    np.save("/nobackup/sclaam/smoothness/trainloss_line_{}_dense.npy".format(name), loss_data_train)
    np.save("/nobackup/sclaam/smoothness/testAccuracy_line_{}_dense.npy".format(name), loss_data_test)
    # np.save("smoothness/testAccuracy_line_{}_dense.npy".format(name), loss_data_test)
    torch.save(best_model.state_dict(), "/nobackup/sclaam/smoothness/{}_stochastic_dense.pth".format(name))
    # torch.save(best_model.state_dict(), "smoothness/{}_stochastic_dense.pth".format(name))
    torch.save(det_pruning_model.state_dict(), "/nobackup/sclaam/smoothness/{}_deterministic_dense.pth".format(name))


def plot_line_(cfg, type_="one_shot"):
    model = get_model(cfg)
    identifier = cfg.id
    # name = "{}_{}_{}_{}_{}".format("test2", cfg.dataset, cfg.architecture, cfg.sigma, cfg.amount)
    name = "{}_{}_{}_{}_{}".format(identifier, cfg.dataset, cfg.architecture, cfg.sigma, cfg.amount)
    train_loss = np.load("smoothness/trainloss_line_{}_{}.npy".format(name, type_))
    stochastic_state_dict = torch.load("smoothness/{}_stochastic_{}.pth".format(name, type_))
    # name = "{}_{}_{}_{}_{}".format(identifier, cfg.dataset, cfg.architecture, cfg.sigma, cfg.amount)
    test_loss = np.load("smoothness/testAccuracy_line_{}_{}.npy".format(name, type_))
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
    train_line = ax.plot(distance_vector, train_loss, color="cornflowerblue", label="Train Loss")
    # ax.set_ylabel(r"$\frac{L}(w)$", fontsize=fs)
    ax.set_ylabel(r"$L(w)$", fontsize=fs)
    ax.set_xlabel("Distance", fontsize=fs)
    # ax.spines['right'].set_color('cornflowerblue')
    ax.tick_params(axis="y", colors="cornflowerblue")

    # plt.legend()
    # plt.show()
    sto_points = []
    det_points = []
    if "one_shot" in type_:
        sto_points.append(
            plt.scatter(distance_vector[stochastic_loss_index], train_loss[stochastic_loss_index], c="cornflowerblue",
                        marker="o", label="Stochastic pruning"))
        det_points.append(plt.scatter(distance_vector[deterministic_loss_index], train_loss[deterministic_loss_index],
                                      c="cornflowerblue",
                                      marker="s", label="Deterministic pruning"))

    elif "dense" in type_:
        sto_points.append(
            plt.scatter(distance_vector[stochastic_loss_index], train_loss[stochastic_loss_index], c="cornflowerblue",
                        marker="o", label="Stochastic"))
        det_points.append(plt.scatter(distance_vector[deterministic_loss_index], train_loss[deterministic_loss_index],
                                      c="cornflowerblue",
                                      marker="s", label="Deterministic"))
    ax2 = plt.twinx()
    test_line = ax2.plot(distance_vector, test_loss, color="limegreen", label="Test Accuracy error")

    if "one_shot" in type_:
        sto_points.append(
            ax2.scatter(distance_vector[stochastic_loss_index], test_loss[stochastic_loss_index], c="limegreen",
                        marker="o", label="Stochastic pruning"))
        det_points.append(
            ax2.scatter(distance_vector[deterministic_loss_index], test_loss[deterministic_loss_index], c="limegreen",
                        marker="s", label="Deterministic pruning"))
    elif "dense" in type_:
        sto_points.append(
            ax2.scatter(distance_vector[stochastic_loss_index], test_loss[stochastic_loss_index], c="limegreen",
                        marker="o", label="Stochastic"))
        det_points.append(
            ax2.scatter(distance_vector[deterministic_loss_index], test_loss[deterministic_loss_index], c="limegreen",
                        marker="s", label="Deterministic"))

    ax2.set_ylabel("Test error ", fontsize=fs)
    # ax2.spines['right'].set_color('limegreen')
    ax2.tick_params(axis="y", colors="limegreen")
    # ax2.yaxis.label.set_color('limegreen')
    det_label = ""
    sto_label = ""
    if "one_shot" in type_:
        sto_label = "Stochastic pruning"
        det_label = "Deterministic pruning"
    elif "dense" in type_:
        sto_label = "Stochastic"
        det_label = "Deterministic"
    sto_handle = mlines.Line2D([0], [0], markerfacecolor='black', marker='o', ls="",
                               markersize=fs * 0.5, label=sto_label)
    det_handle = mlines.Line2D([0], [0], markerfacecolor='black', marker='s', ls="",
                               markersize=fs * 0.5, label=det_label)
    legend_list = plt.legend([tuple(train_line), tuple(test_line), sto_handle, det_handle],
                             ["Train loss", "Test Error", sto_label, det_label], loc="upper left",
                             scatterpoints=1, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)},
                             prop={"size": fs * 0.8})
    legend_list.legend_handles[2].set_color('black')
    # legend_list.legend_handles[2].set_facecolor('black')
    legend_list.legend_handles[3].set_color('black')
    ax2.set_ylim(np.min(test_loss) - 5, 100)
    # legend_list.legend_handles[3].set_facecolor('black')
    plt.grid(ls='--', alpha=0.5)
    plt.savefig("paper_plots/line_{}_{}.pdf".format(name, type_))

    fig, ax = plt.subplots(figsize=fig_size, layout="compressed")
    train_line = ax.plot(distance_vector[50:150], train_loss[50:150], color="cornflowerblue", label="Train Loss")
    ax.tick_params(axis="y", colors="cornflowerblue")
    ax.set_ylabel(r"$L(w)$", fontsize=fs)
    ax.set_xlabel("Distance", fontsize=fs)
    sto_points = []
    det_points = []

    if "one_shot" in type_:
        sto_points.append(
            ax.scatter(distance_vector[stochastic_loss_index], train_loss[stochastic_loss_index], c="cornflowerblue",
                       marker="o", label="Stochastic pruning"))
        det_points.append(
            ax.scatter(distance_vector[deterministic_loss_index], train_loss[deterministic_loss_index],
                       c="cornflowerblue",
                       marker="s", label="Deterministic pruning"))
    elif "dense" in type_:
        sto_points.append(
            ax.scatter(distance_vector[stochastic_loss_index], train_loss[stochastic_loss_index], c="cornflowerblue",
                       marker="o", label="Stochastic"))
        det_points.append(
            ax.scatter(distance_vector[deterministic_loss_index], train_loss[deterministic_loss_index],
                       c="cornflowerblue",
                       marker="s", label="Deterministic"))
    ax2 = plt.twinx()
    test_line = ax2.plot(distance_vector[50:150], test_loss[50:150], color="limegreen", label="Test Accuracy error")
    if "one_shot" in type_:
        sto_points.append(
            ax2.scatter(distance_vector[stochastic_loss_index], test_loss[stochastic_loss_index], c="limegreen",
                        marker="o", label="Stochastic pruning"))
        det_points.append(
            ax2.scatter(distance_vector[deterministic_loss_index], test_loss[deterministic_loss_index], c="limegreen",
                        marker="s", label="Deterministic pruning"))
    elif "dense" in type_:
        sto_points.append(
            ax2.scatter(distance_vector[stochastic_loss_index], test_loss[stochastic_loss_index], c="limegreen",
                        marker="o", label="Stochastic"))
        det_points.append(
            ax2.scatter(distance_vector[deterministic_loss_index], test_loss[deterministic_loss_index], c="limegreen",
                        marker="s", label="Deterministic"))
    det_label = ""
    sto_label = ""
    if "one_shot" in type_:
        sto_label = "Stochastic pruning"
        det_label = "Deterministic pruning"
    elif "dense" in type_:
        sto_label = "Stochastic"
        det_label = "Deterministic"
    sto_handle = mlines.Line2D([0], [0], markerfacecolor='black', marker='o', linestyle=None, ls='',
                               markersize=fs * 0.5, markeredgecolor="k", label=sto_label)
    det_handle = mlines.Line2D([0], [0], markerfacecolor='black', marker='s', ls='',
                               markersize=fs * 0.5, markeredgecolor="k", linestyle=None, label=det_label)
    # handle_=
    legend_list = plt.legend([tuple(train_line), tuple(test_line), sto_handle, det_handle],
                             ["Train loss", "Test Error", sto_label, det_label], loc="upper left",
                             scatterpoints=1, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)},
                             prop={"size": fs * 0.8})

    legend_list.legend_handles[2].set_color('black')
    # legend_list.legend_handles[2].set_linestyle('black')
    # legend_list.legend_handles[2].set_facecolor('black')
    legend_list.legend_handles[3].set_color('black')
    # legend_list.legend_handles[3].set_facecolor('black')

    ax2.tick_params(axis="y", colors="limegreen")
    ax2.set_ylabel("Test error ", fontsize=fs)
    # ax2.set_ylim(20,100)
    ax2.set_ylim(np.min(test_loss) - 5, 100)
    # plt.savefig("paper_plots/line_{}_{}.pdf".format(name, type_))
    plt.grid(ls='--', alpha=0.5)
    plt.savefig("paper_plots/zoom_line_{}_{}.pdf".format(name, type_))


def plot_left_line_(cfg, type_="one_shot"):
    model = get_model(cfg)
    identifier = cfg.id
    # name = "{}_{}_{}_{}_{}".format("test2", cfg.dataset, cfg.architecture, cfg.sigma, cfg.amount)
    name = "{}_{}_{}_{}_{}".format(identifier, cfg.dataset, cfg.architecture, cfg.sigma, cfg.amount)
    dense_train_loss = np.load("smoothness/trainloss_left_line_{}_dense.npy".format(name, type_))
    sparse_train_loss = np.load("smoothness/trainloss_left_line_{}_sparse.npy".format(name, type_))
    stochastic_state_dict = torch.load("smoothness/{}_left_stochastic_dense.pth".format(name, type_))
    # name = "{}_{}_{}_{}_{}".format(identifier, cfg.dataset, cfg.architecture, cfg.sigma, cfg.amount)
    dense_test_loss = np.load("smoothness/testAccuracy_left_line_{}_dense.npy".format(name, type_))
    sparse_test_loss = np.load("smoothness/testAccuracy_left_line_{}_sparse.npy".format(name, type_))
    deterministic_state_dict = torch.load("smoothness/{}_left_deterministic_dense.pth".format(name))
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
    dense_train_line = ax.plot(distance_vector, dense_train_loss, color="cornflowerblue", label="Dense Train Loss")
    sparse_train_line = ax.plot(distance_vector, sparse_train_loss, color="limegreen", label="SparseTrain Loss")
    # ax.set_ylabel(r"$\frac{L}(w)$", fontsize=fs)
    ax.set_ylabel(r"$L(w)$", fontsize=fs)
    ax.set_xlabel("Distance", fontsize=fs)
    # ax.spines['right'].set_color('cornflowerblue')
    ax.tick_params(axis="y", colors="cornflowerblue")

    # plt.legend()
    # plt.show()
    sto_points = []
    det_points = []

    sto_points.append(
        plt.scatter(distance_vector[stochastic_loss_index], sparse_train_loss[stochastic_loss_index], c="limegreen",
                    marker="o", label="Stochastic pruning"))
    det_points.append(
        plt.scatter(distance_vector[deterministic_loss_index], sparse_train_loss[deterministic_loss_index],
                    c="limegreen",
                    marker="s", label="Deterministic pruning"))

    sto_points.append(
        plt.scatter(distance_vector[stochastic_loss_index], dense_train_loss[stochastic_loss_index], c="cornflowerblue",
                    marker="o", label="Stochastic"))
    det_points.append(plt.scatter(distance_vector[deterministic_loss_index], dense_train_loss[deterministic_loss_index],
                                  c="cornflowerblue",
                                  marker="s", label="Deterministic"))

    det_label = ""
    sto_label = ""
    if "one_shot" in type_:
        sto_label = "Stochastic pruning"
        det_label = "Deterministic pruning"
    elif "dense" in type_:
        sto_label = "Stochastic"
        det_label = "Deterministic"
    sto_handle = mlines.Line2D([0], [0], markerfacecolor='black', marker='o', ls="",
                               markersize=fs * 0.5, label=sto_label)
    det_handle = mlines.Line2D([0], [0], markerfacecolor='black', marker='s', ls="",
                               markersize=fs * 0.5, label=det_label)
    legend_list = plt.legend([tuple(dense_train_line), tuple(sparse_train_line), sto_handle, det_handle],
                             ["Dense Landscape", "Sparse Landscape", sto_label, det_label], loc="upper left",
                             scatterpoints=1, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)},
                             prop={"size": fs * 0.8})
    legend_list.legend_handles[2].set_color('black')
    # legend_list.legend_handles[2].set_facecolor('black')
    legend_list.legend_handles[3].set_color('black')
    # ax2.set_ylim(np.min(test_loss) - 5, 100)
    # legend_list.legend_handles[3].set_facecolor('black')
    plt.grid(ls='--', alpha=0.5)
    plt.savefig("paper_plots/train_line_dense_sparse_{}_{}.pdf".format(name, type_))

    fig, ax2 = plt.subplots(figsize=fig_size, layout="compressed")

    # ax2 = plt.twinx()
    dense_test_line = ax2.plot(distance_vector, dense_test_loss, c="cornflowerblue", label="Test Accuracy error")
    sparse_test_line = ax2.plot(distance_vector, sparse_test_loss, color="limegreen", label="Test Accuracy error")

    sto_points.append(
        ax2.scatter(distance_vector[stochastic_loss_index], sparse_test_loss[stochastic_loss_index], c="limegreen",
                    marker="o", label="Stochastic pruning"))
    det_points.append(
        ax2.scatter(distance_vector[deterministic_loss_index], sparse_test_loss[deterministic_loss_index],
                    c="limegreen",
                    marker="s", label="Deterministic pruning"))
    sto_points.append(
        ax2.scatter(distance_vector[stochastic_loss_index], dense_test_loss[stochastic_loss_index], c="cornflowerblue",
                    marker="o", label="Stochastic"))
    det_points.append(
        ax2.scatter(distance_vector[deterministic_loss_index], dense_test_loss[deterministic_loss_index],
                    c="cornflowerblue",
                    marker="s", label="Deterministic"))

    ax2.set_ylabel("Test error ", fontsize=fs)
    # ax2.spines['right'].set_color('limegreen')
    ax2.tick_params(axis="y", colors="limegreen")
    # ax2.yaxis.label.set_color('limegreen')
    det_label = ""
    sto_label = ""
    if "one_shot" in type_:
        sto_label = "Stochastic pruning"
        det_label = "Deterministic pruning"
    elif "dense" in type_:
        sto_label = "Stochastic"
        det_label = "Deterministic"
    sto_handle = mlines.Line2D([0], [0], markerfacecolor='black', marker='o', ls="",
                               markersize=fs * 0.5, label=sto_label)
    det_handle = mlines.Line2D([0], [0], markerfacecolor='black', marker='s', ls="",
                               markersize=fs * 0.5, label=det_label)
    legend_list = plt.legend([tuple(dense_test_line), tuple(sparse_test_line), sto_handle, det_handle],
                             ["Dense Test error", "Sparse Test error", sto_label, det_label], loc="upper left",
                             scatterpoints=1, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)},
                             prop={"size": fs * 0.8})
    legend_list.legend_handles[2].set_color('black')
    # legend_list.legend_handles[2].set_facecolor('black')
    legend_list.legend_handles[3].set_color('black')
    # ax2.set_ylim(np.min(test_loss) - 5, 100)
    # legend_list.legend_handles[3].set_facecolor('black')
    plt.grid(ls='--', alpha=0.5)
    plt.savefig("paper_plots/test_line_sparse_dense_{}_{}.pdf".format(name, type_))


    ##################################################
    #                        zoom
    ##################################################


    fig, ax = plt.subplots(figsize=fig_size, layout="compressed")
    dense_train_line = ax.plot(distance_vector[50:150], dense_train_loss[50:150], color="cornflowerblue", label="Dense Train Loss")
    sparse_train_line = ax.plot(distance_vector[50:150], sparse_train_loss[50:150], color="limegreen", label="SparseTrain Loss")
    # ax.set_ylabel(r"$\frac{L}(w)$", fontsize=fs)
    ax.set_ylabel(r"$L(w)$", fontsize=fs)
    ax.set_xlabel("Distance", fontsize=fs)
    # ax.spines['right'].set_color('cornflowerblue')
    ax.tick_params(axis="y", colors="cornflowerblue")

    # plt.legend()
    # plt.show()
    sto_points = []
    det_points = []

    sto_points.append(
        plt.scatter(distance_vector[stochastic_loss_index], sparse_train_loss[stochastic_loss_index], c="limegreen",
                    marker="o", label="Stochastic pruning"))
    det_points.append(
        plt.scatter(distance_vector[deterministic_loss_index], sparse_train_loss[deterministic_loss_index],
                    c="limegreen",
                    marker="s", label="Deterministic pruning"))

    sto_points.append(
        plt.scatter(distance_vector[stochastic_loss_index], dense_train_loss[stochastic_loss_index], c="cornflowerblue",
                    marker="o", label="Stochastic"))
    det_points.append(plt.scatter(distance_vector[deterministic_loss_index], dense_train_loss[deterministic_loss_index],
                                  c="cornflowerblue",
                                  marker="s", label="Deterministic"))

    det_label = ""
    sto_label = ""
    if "one_shot" in type_:
        sto_label = "Stochastic pruning"
        det_label = "Deterministic pruning"
    elif "dense" in type_:
        sto_label = "Stochastic"
        det_label = "Deterministic"
    sto_handle = mlines.Line2D([0], [0], markerfacecolor='black', marker='o', ls="",
                               markersize=fs * 0.5, label=sto_label)
    det_handle = mlines.Line2D([0], [0], markerfacecolor='black', marker='s', ls="",
                               markersize=fs * 0.5, label=det_label)
    legend_list = plt.legend([tuple(dense_train_line), tuple(sparse_train_line), sto_handle, det_handle],
                             ["Dense Landscape", "Sparse Landscape", sto_label, det_label], loc="upper left",
                             scatterpoints=1, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)},
                             prop={"size": fs * 0.8})
    legend_list.legend_handles[2].set_color('black')
    # legend_list.legend_handles[2].set_facecolor('black')
    legend_list.legend_handles[3].set_color('black')
    # ax2.set_ylim(np.min(test_loss) - 5, 100)
    # legend_list.legend_handles[3].set_facecolor('black')
    plt.grid(ls='--', alpha=0.5)
    plt.savefig("paper_plots/zoom_train_line_dense_sparse_{}_{}.pdf".format(name, type_))

    fig, ax2 = plt.subplots(figsize=fig_size, layout="compressed")

    # ax2 = plt.twinx()
    dense_test_line = ax2.plot(distance_vector[50:150], dense_test_loss[50:150], c="cornflowerblue", label="Test Accuracy error")
    sparse_test_line = ax2.plot(distance_vector[50:150], sparse_test_loss[50:150], color="limegreen", label="Test Accuracy error")

    sto_points.append(
        ax2.scatter(distance_vector[stochastic_loss_index], sparse_test_loss[stochastic_loss_index], c="limegreen",
                    marker="o", label="Stochastic pruning"))
    det_points.append(
        ax2.scatter(distance_vector[deterministic_loss_index], sparse_test_loss[deterministic_loss_index],
                    c="limegreen",
                    marker="s", label="Deterministic pruning"))
    sto_points.append(
        ax2.scatter(distance_vector[stochastic_loss_index], dense_test_loss[stochastic_loss_index], c="cornflowerblue",
                    marker="o", label="Stochastic"))
    det_points.append(
        ax2.scatter(distance_vector[deterministic_loss_index], dense_test_loss[deterministic_loss_index],
                    c="cornflowerblue",
                    marker="s", label="Deterministic"))

    ax2.set_ylabel("Test error ", fontsize=fs)
    # ax2.spines['right'].set_color('limegreen')
    ax2.tick_params(axis="y", colors="limegreen")
    # ax2.yaxis.label.set_color('limegreen')
    det_label = ""
    sto_label = ""
    if "one_shot" in type_:
        sto_label = "Stochastic pruning"
        det_label = "Deterministic pruning"
    elif "dense" in type_:
        sto_label = "Stochastic"
        det_label = "Deterministic"
    sto_handle = mlines.Line2D([0], [0], markerfacecolor='black', marker='o', ls="",
                               markersize=fs * 0.5, label=sto_label)
    det_handle = mlines.Line2D([0], [0], markerfacecolor='black', marker='s', ls="",
                               markersize=fs * 0.5, label=det_label)
    legend_list = plt.legend([tuple(dense_test_line), tuple(sparse_test_line), sto_handle, det_handle],
                             ["Dense Test error", "Sparse Test error", sto_label, det_label], loc="upper left",
                             scatterpoints=1, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=1)},
                             prop={"size": fs * 0.8})
    legend_list.legend_handles[2].set_color('black')
    # legend_list.legend_handles[2].set_facecolor('black')
    legend_list.legend_handles[3].set_color('black')
    # ax2.set_ylim(np.min(test_loss) - 5, 100)
    # legend_list.legend_handles[3].set_facecolor('black')
    plt.grid(ls='--', alpha=0.5)
    plt.savefig("paper_plots/zoom_test_line_sparse_dense_{}_{}.pdf".format(name, type_))


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
    # linear_interpolation_oneshot_GMP(cfg)
    # linear_interpolation_dense_GMP(cfg)
    # left_pruning_experiments(cfg)
    plot_left_line_(cfg, args["type"])
