import logging
import typing
from functools import partial
from typing import TYPE_CHECKING
import omegaconf
# from main import get_layer_dict
import optuna.samplers
from einops import repeat
import numpy as np
import torch
from torch import nn
from shrinkbench.metrics.flops import flops
from torchmetrics import Accuracy
import wandb
from decimal import Decimal

def test(net, use_cuda, testloader, one_batch=False, verbose=2, count_flops=False, batch_flops=0):
    if use_cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

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
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % 100 == 0:
                if verbose == 2:
                    print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                          % (test_loss / (batch_idx + 1), 100. * correct.item() / total, correct, total))
            if one_batch:
                return 100. * correct.item() / total
    if verbose == 1 or verbose == 2:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss / len(testloader), correct, total,
            100. * correct.item() / total))
    net.cpu()
    if count_flops:
        return 100. * correct.item() / total, sparse_flops
    else:
        return 100. * correct.item() / total


def get_mask(weight: torch.FloatTensor):
    return (weight != 0).type(torch.float)


def mask_gradient(model: torch.nn.Module, mask_dict: dict):
    parameters_dict = dict(model.named_parameters())
    # for name,parameter in parameters_dict.items():
    #     for mask_name in mask_dict.keys():
    #         if mask_name in name:
    #             parameter.grad.data.mul_(mask_dict[mask_name].to("cuda"))
    for name, module in model.named_modules():
        if name in mask_dict.keys():
            if hasattr(module.weight, "grad"):
                # print("Module Name: {}".format(name))
                module.weight.grad.mul_(mask_dict[name].to("cuda"))


def restricted_fine_tune_measure_flops(pruned_model: nn.Module, dataLoader: torch.utils.data.DataLoader,
                                       testLoader: torch.utils.data.DataLoader,
                                       epochs=1,
                                       FLOP_limit: float = 0, use_wandb=False):
    # optimizer = torch.optim.SGD()
    optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.0001,
                                momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    grad_clip = 0.1
    names, weights = zip(*get_layer_dict(pruned_model))
    accuracy = Accuracy(task="multiclass", num_classes=10).to("cuda")

    masks = list(map(get_mask, weights))
    mask_dict = dict(zip(names, masks))
    criterion = nn.CrossEntropyLoss()
    total_FLOPS = 0
    total_sparse_FLOPS = 0
    # first_time = 1
    forward_pass_sparse_flops = 0
    forward_pass_dense_flops = 0
    data, y = next(iter(dataLoader))
    forward_pass_dense_flops, forward_pass_sparse_flops = flops(pruned_model, data)
    pruned_model.cuda()
    pruned_model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataLoader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # first forward-backward step
            predictions = pruned_model(data)
            # enable_bn(model)
            loss = criterion(predictions, target)
            loss.backward()
            backward_flops_sparse = 2 * forward_pass_sparse_flops
            backward_flops_dense = 2 * forward_pass_dense_flops
            batch_dense_flops = forward_pass_dense_flops + backward_flops_dense
            batch_sparse_flops = forward_pass_sparse_flops + backward_flops_sparse
            total_FLOPS += batch_dense_flops
            total_sparse_FLOPS += batch_sparse_flops
            accuracy.update(preds=predictions.cuda(), target=target.cuda())
            mask_gradient(pruned_model, mask_dict=mask_dict)

            if grad_clip:
                nn.utils.clip_grad_value_(pruned_model.parameters(), grad_clip)

            optimizer.step()
            lr_scheduler.step()
            if use_wandb:
                acc = accuracy.compute()
                wandb.log({
                    "val_set_accuracy": acc,
                    "sparse_flops": total_sparse_FLOPS,
                })
            if batch_idx % 10 == 0 or FLOP_limit != 0:
                acc = accuracy.compute()
                flops_sparse = '%.3E' % Decimal(total_sparse_FLOPS)
                print(f"Fine-tune Results - Epoch: {epoch}  Avg accuracy: {acc:.2f} Avg loss:"
                      f" {loss.item():.2f} FLOPS:{flops_sparse} sparsity {sparstiy(pruned_model):.3f}")

                if FLOP_limit != 0 and FLOP_limit > total_sparse_FLOPS:
                    break
        if FLOP_limit != 0:
            if total_sparse_FLOPS > FLOP_limit:
                break
    test_set_performance = test(pruned_model, use_cuda=True, testloader=testLoader)

    if use_wandb:
        wandb.log({
            "test_set_accuracy": test_set_performance
        })
        wandb.join()
    # msg_perormance = f"{performance:.2f}".replace(".", ",")


#
# from .counting.ops import get_inference_FLOPs
#
# # Sparse learning funcs
# from .funcs.grow import registry as grow_registry
# from .funcs.init_scheme import registry as init_registry
# from .funcs.prune import registry as prune_registry
# from .funcs.redistribute import registry as redistribute_registry
# from .sparse_ensemble_utils.smoothen_value import AverageValue
def is_prunable_module(m: torch.nn.Module):
    return (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d))


def get_sampler(cfg: omegaconf.DictConfig):
    if cfg.sampler == "tpe":
        return optuna.samplers.TPESampler()
    if cfg.sampler == "cmaes":
        return optuna.samplers.CmaEsSampler(
            restart_strategy="ipop",
            inc_popsize=2,
            n_startup_trials=10

        )
    if cfg.sampler == "qmc":
        return optuna.samplers.QMCSampler()
    raise NotImplementedError("Sampler {} is not supported yet".format(cfg.sampler))


def get_layer_dict(model: torch.nn.Module):
    """
    :param model:
    :return: The name of the modules that are not batch norm and their correspondent weight with original shape.
    """
    iter_1 = model.named_modules()
    layer_dict = []

    for name, m in iter_1:
        with torch.no_grad():
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not \
                    isinstance(m, nn.BatchNorm3d):
                layer_dict.append((name, m.weight.data.cpu().detach()))
                # if "shortcut" in name:
                #     print(f"{name}:{m.weight.data}")
                # if not isinstance(m, nn.Conv2d):
                #     print(f"{name}:{m.weight.data}")

    return layer_dict


# Function taken from https://github.com/varun19299/rigl-reproducibility/blob/master/sparselearning/utils/ops.py
def random_perm(a: torch.Tensor) -> torch.Tensor:
    """
    Random shuffle a tensor.

    :param a: input Tensor
    :type a: torch.Tensor
    :return: shuffled Tensor
    :rtype: torch.Tensor
    """
    idx = torch.randperm(a.nelement())
    return a.reshape(-1)[idx].reshape(a.shape)


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def get_percentile_per_layer(model: torch.nn.Module, percentile: float = 0.1):
    return_dict = {}
    for name, module in model.named_modules():
        if is_prunable_module(module):
            flat_weights = torch.abs(module.weight.detach().flatten())
            desired_quantile = float(torch.quantile(flat_weights, percentile))
            return_dict[name] = desired_quantile
    return return_dict


def sparstiy(model):
    # check if they hav
    total_params = count_parameters(model)
    non_zero_param = 0
    for name, module in model.named_modules():
        if is_prunable_module(module):

            if list(module.buffers()):
                list_buffers = list(module.buffers())
                non_zero_param += len(list_buffers[0].flatten().nonzero())
            else:

                non_zero_param += len(module.weight.data.flatten().nonzero())
    return 1 - non_zero_param / total_params


# Functions adapted from https://github.com/varun19299/rigl-reproducibility/blob/master/sparselearning/funcs
# /init_scheme.py


def erdos_renyi_per_layer_pruning_rate(model: torch.nn.Module, cfg: omegaconf.DictConfig, is_kernel:
bool = True,
                                       **kwargs) -> typing.Tuple[dict, dict, float, int]:
    names, weights = zip(*get_layer_dict(model))
    prob_dict = get_erdos_renyi_dist(model, names, weights, cfg, is_kernel)
    mask_dict = {}
    pruning_rate_per_layer = {}
    baseline_nonzero = 0
    total_params = 0
    for name, weight in zip(names, weights):
        if name in cfg.exclude_layers:
            continue
        prob = prob_dict[name]
        logging.debug(f"ERK {name}: {weight.shape} prob {prob:.4f}")

        mask_dict[name] = (torch.rand(weight.shape) < prob).float().data
        pruning_rate_per_layer[name] = ((mask_dict[name] == 0).sum() / mask_dict[name].nelement()).item()
        baseline_nonzero += (mask_dict[name] != 0).sum().int().item()
        total_params += weight.numel()

    return mask_dict, pruning_rate_per_layer, baseline_nonzero, total_params


def get_erdos_renyi_dist(
        model, names, weights, cfg: omegaconf.DictConfig, is_kernel: bool = True
) -> "typing.Dict[str, float]":
    """
    Get layer-wise densities distributed according to
    ER or ERK (erdos-renyi or erdos-renyi-kernel).

    Ensures resulting densities do not cross 1
    for any layer.

    :param masking: Masking instance
    :param is_kernel: use ERK (True), ER (False)
    :return: Layer-wise density dict
    """
    # Same as Erdos Renyi with modification for conv
    # initialization used in sparse evolutionary training
    # scales the number of non-zero weights linearly proportional
    # to the product of all dimensions, that is input*output
    # for fully connected layers, and h*w*in_c*out_c for conv
    # layers.
    _erk_power_scale = 1.0

    epsilon = 1.0
    is_epsilon_valid = False
    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    _dense_layers = set()
    while not is_epsilon_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.
        # We want the total number of connections to be the same. Let say we have
        # for layers with N_1, ..., N_4 parameters each. Let say after some
        # iterations probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. See below.
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, weight in zip(names, weights):
            n_param = np.prod(weight.shape)
            n_zeros = int(n_param * (cfg.amount))
            n_ones = int(n_param * (1 - cfg.amount))

            if name in _dense_layers:
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros

            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                rhs += n_ones

                if is_kernel:
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (
                                                      np.sum(weight.shape) / np.prod(weight.shape)
                                              ) ** _erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                else:
                    # Cin and Cout for a conv kernel
                    n_in, n_out = weight.shape[:2]
                    raw_probabilities[name] = (n_in + n_out) / (n_in * n_out)
                divisor += raw_probabilities[name] * n_param
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        epsilon = rhs / divisor
        # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    logging.info(f"Density of layer:{mask_name} set to 1.0")
                    _dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    prob_dict = {}
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for name, module in model.named_modules():
        if name in cfg.exclude_layers or name not in names:
            continue
        if name in _dense_layers:
            prob = 1.0
        else:
            prob = epsilon * raw_probabilities[name]

        prob_dict[name] = prob

    return prob_dict
