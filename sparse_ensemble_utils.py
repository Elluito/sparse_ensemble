import logging
import typing
from functools import partial
from typing import TYPE_CHECKING
import omegaconf
# from main import get_layer_dict
from einops import repeat
import numpy as np
import torch
from torch import nn


#
# from .counting.ops import get_inference_FLOPs
#
# # Sparse learning funcs
# from .funcs.grow import registry as grow_registry
# from .funcs.init_scheme import registry as init_registry
# from .funcs.prune import registry as prune_registry
# from .funcs.redistribute import registry as redistribute_registry
# from .sparse_ensemble_utils.smoothen_value import AverageValue

def get_layer_dict(model):
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


# Functions adapted from https://github.com/varun19299/rigl-reproducibility/blob/master/sparselearning/funcs
# /init_scheme.py

def erdos_renyi_per_layer_pruning_rate(model: torch.nn.Module, cfg: omegaconf.DictConfig, is_kernel: bool = True,
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
