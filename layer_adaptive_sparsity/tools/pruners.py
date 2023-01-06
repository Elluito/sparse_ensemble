import torch
from torch.nn.utils import prune
from .utils import get_weights, get_modules
import numpy as np


def weight_pruner_loader(pruner_string):
    """
    Gives you the pruning methods: LAMP, Glob, Unif, Unif+, and ERK
    """
    if pruner_string == 'lamp':
        return prune_weights_lamp
    elif pruner_string == 'glob':
        return prune_weights_global
    elif pruner_string == 'unif':
        return prune_weights_uniform
    elif pruner_string == 'unifplus':
        return prune_weights_unifplus
    elif pruner_string == 'erk':
        return prune_weights_erk
    else:
        raise ValueError('Unknown pruner')


"""
prune_weights_reparam: Allocate identity mask to every weight tensors.
prune_weights_l1predefined: Perform layerwise pruning w.r.t. given amounts.
"""


def prune_weights_reparam(model):
    module_list = get_modules(model)
    for m in module_list.values():
        prune.identity(m, name="weight")


def prune_weights_l1predefined(model, amounts, exclude_layers=[]):
    mdict = get_modules(model, exclude_layers)
    for name, m in mdict.items():
        prune.l1_unstructured(m, name="weight", amount=float(amounts[name]))


"""
Methods: All weights
"""


def prune_weights_global(model, amount):
    parameters_to_prune = _extract_weight_tuples(model)
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)


def prune_weights_lamp(model, amount, exclude_layers: list = []):
    assert amount <= 1
    amounts = _compute_lamp_amounts(model, amount, exclude_layers=exclude_layers)
    prune_weights_l1predefined(model, amounts,exclude_layers=exclude_layers)


def prune_weights_uniform(model, amount):
    module_list = get_modules(model)
    assert amount <= 1  # Can be updated later to handle > 1.
    for m in module_list:
        prune.l1_unstructured(m, name="weight", amount=amount)


def prune_weights_unifplus(model, amount):
    assert amount <= 1
    amounts = _compute_unifplus_amounts(model, amount)
    prune_weights_l1predefined(model, amounts)


def prune_weights_erk(model, amount):
    assert amount <= 1
    amounts = _compute_erk_amounts(model, amount)
    prune_weights_l1predefined(model, amounts)


"""
These are not intended to be exported.
"""


def _extract_weight_tuples(model):
    """
    Gives you well-packed weight tensors for global pruning.
    """
    mlist = get_modules(model)
    return tuple([(m, 'weight') for m in mlist])


def _compute_unifplus_amounts(model, amount):
    """
    Compute # of weights to prune in each layer.
    """
    amounts = []
    wlist = get_weights(model)
    unmaskeds = _count_unmasked_weights(model)
    totals = _count_total_weights(model)

    last_layer_minimum = np.round(totals[-1] * 0.2)  # Minimum number of last-layer weights to keep
    total_to_prune = np.round(unmaskeds.sum() * amount)

    if wlist[0].dim() == 4:
        amounts.append(0)  # Leave the first layer unpruned.
        frac_to_prune = (total_to_prune * 1.0) / (unmaskeds[1:].sum())
        if frac_to_prune > 1.0:
            raise ValueError("Cannot be pruned further by the Unif+ scheme! (first layer exception)")
        last_layer_to_surv_planned = np.round((1.0 - frac_to_prune) * unmaskeds[-1])
        if last_layer_to_surv_planned < last_layer_minimum:
            last_layer_to_prune = unmaskeds[-1] - last_layer_minimum
            frac_to_prune_middle = ((total_to_prune - last_layer_to_prune) * 1.0) / (unmaskeds[1:-1].sum())
            if frac_to_prune_middle > 1.0:
                raise ValueError("Cannot be pruned further by the Unif+ scheme! (first+last layer exception)")
            amounts.extend([frac_to_prune_middle] * (unmaskeds.size(0) - 2))
            amounts.append((last_layer_to_prune * 1.0) / unmaskeds[-1])
        else:
            amounts.extend([frac_to_prune] * (unmaskeds.size(0) - 1))
    else:
        frac_to_prune = (total_to_prune * 1.0) / (unmaskeds.sum())
        last_layer_to_surv_planned = np.round((1.0 - frac_to_prune) * unmaskeds[-1])
        if last_layer_to_surv_planned < last_layer_minimum:
            last_layer_to_prune = unmaskeds[-1] - last_layer_minimum
            frac_to_prune_middle = ((total_to_prune - last_layer_to_prune) * 1.0) / (unmaskeds[:-1].sum())
            if frac_to_prune_middle > 1.0:
                raise ValueError("Cannot be pruned further by the Unif+ scheme! (last layer exception)")
            amounts.extend([frac_to_prune_middle] * (unmaskeds.size(0) - 1))
            amounts.append((last_layer_to_prune * 1.0) / unmaskeds[-1])
        else:
            amounts.extend([frac_to_prune] * (unmaskeds.size(0)))
    return amounts


def _compute_erk_amounts(model, amount):
    unmaskeds = _count_unmasked_weights(model)
    erks = _compute_erks(model)

    return _amounts_from_eps(unmaskeds, erks, amount)


def _amounts_from_eps(unmaskeds, ers, amount):
    num_layers = ers.size(0)
    layers_to_keep_dense = torch.zeros(num_layers)
    total_to_survive = (1.0 - amount) * unmaskeds.sum()  # Total to keep.

    # Determine some layers to keep dense.
    is_eps_invalid = True
    while is_eps_invalid:
        unmasked_among_prunables = (unmaskeds * (1 - layers_to_keep_dense)).sum()
        to_survive_among_prunables = total_to_survive - (layers_to_keep_dense * unmaskeds).sum()

        ers_of_prunables = ers * (1.0 - layers_to_keep_dense)
        survs_of_prunables = torch.round(to_survive_among_prunables * ers_of_prunables / ers_of_prunables.sum())

        layer_to_make_dense = -1
        max_ratio = 1.0
        for idx in range(num_layers):
            if layers_to_keep_dense[idx] == 0:
                if survs_of_prunables[idx] / unmaskeds[idx] > max_ratio:
                    layer_to_make_dense = idx
                    max_ratio = survs_of_prunables[idx] / unmaskeds[idx]

        if layer_to_make_dense == -1:
            is_eps_invalid = False
        else:
            layers_to_keep_dense[layer_to_make_dense] = 1

    amounts = torch.zeros(num_layers)

    for idx in range(num_layers):
        if layers_to_keep_dense[idx] == 1:
            amounts[idx] = 0.0
        else:
            amounts[idx] = 1.0 - (survs_of_prunables[idx] / unmaskeds[idx])
    return amounts


def _compute_lamp_amounts(model, amount, exclude_layers):
    """
    Compute normalization schemes.
    """
    unmaskeds_dict = _count_unmasked_weights(model, exclude_layers)
    unmaskeds = torch.FloatTensor(list(unmaskeds_dict.values()))
    num_surv = int(np.round(unmaskeds.sum() * (1.0 - amount)))

    flattened_scores_dict = dict([(name, _normalize_scores(w ** 2).view(-1)) for name, w in get_weights(model,
                                                                                                        exclude_layers).items()])

    concat_scores = torch.cat(tuple(flattened_scores_dict.values()), dim=0)
    topks, _ = torch.topk(concat_scores, num_surv)
    threshold = topks[-1]

    # We don't care much about tiebreakers, for now.
    final_survs = [(name, torch.ge(score, threshold * torch.ones(score.size()).to(score.device)).sum()) for name,
                                                                                                            score in
                   flattened_scores_dict.items()]
    amounts = []
    for name, final_surv in final_survs:
        amounts.append((name, (1.0 - (final_surv / unmaskeds_dict[name]))))

    return dict(amounts)


def _compute_erks(model):
    wlist = get_weights(model)
    erks = torch.zeros(len(wlist))
    for idx, w in enumerate(wlist):
        if w.dim() == 4:
            erks[idx] = w.size(0) + w.size(1) + w.size(2) + w.size(3)
        else:
            erks[idx] = w.size(0) + w.size(1)
    return erks


def _count_unmasked_weights(model, exclude_layers=[]):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist = get_modules(model, exclude_layers=exclude_layers)
    unmaskeds = []
    for name, m in mlist.items():
        if hasattr(m, "weight_mask"):
            unmaskeds.append((name, m.weight_mask.sum()))
        else:
            unmaskeds.append((name, torch.count_nonzero(m.weight.data)))
    return dict(unmaskeds)


def _count_total_weights(model):
    """
    Return a 1-dimensional tensor of #total weights.
    """
    wlist = get_weights(model)
    numels = []
    for w in wlist:
        numels.append(w.numel())
    return torch.FloatTensor(numels)


def _normalize_scores(scores):
    """
    Normalizing scheme for LAMP.
    """
    # sort scores in an ascending order
    sorted_scores, sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp) - 1]
    # normalize by cumulative sum
    sorted_scores /= (scores.sum() - scores_cumsum)
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape, device=scores.device)
    new_scores[sorted_idx] = sorted_scores

    return new_scores.view(scores.shape)
