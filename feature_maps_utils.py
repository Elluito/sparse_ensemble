import functools
from collections import OrderedDict, defaultdict
from typing import List, Any
import numpy as np
from pathlib import Path
import torch
from torch import nn


def hook_applyfn(hook, model, forward=False, backward=False):
    """

    [description]

    Arguments:
        hook {[type]} -- [description]
        model {[type]} -- [description]

    Keyword Arguments:
        forward {bool} -- [description] (default: {False})
        backward {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    assert forward ^ backward, \
        "Either forward or backward must be True"
    hooks: list[Any] = []

    def register_hook(module):
        if (
                not isinstance(module, nn.Sequential)
                and
                not isinstance(module, nn.ModuleList)
                and
                not isinstance(module, nn.ModuleDict)
                and
                not (module == model)
        ):
            if forward:
                hooks.append(module.register_forward_hook(hook))
            if backward:
                hooks.append(module.register_backward_hook(hook))

    return register_hook, hooks


def hook_apply_partial_fn(hook, model, forward=False, backward=False):
    assert forward ^ backward, \
        "Either forward or backward must be True"
    hooks = []

    def register_hook(module, name):
        if (
                not isinstance(module, nn.Sequential)
                and
                not isinstance(module, nn.ModuleList)
                and
                not isinstance(module, nn.ModuleDict)
                and
                not (module == model)
        ):
            if forward:
                hooks.append(module.register_forward_hook(functools.partial(hook, name)))
            if backward:
                hooks.append(module.register_backward_hook(hook))

    return register_hook, hooks


def get_activations_shape(model, input):
    # activations = OrderedDict()
    inputs = []
    activations = []
    module_names = []

    def store_activations(module, input, output):
        if isinstance(module, nn.ReLU):
            # TODO ResNet18 implementation reuses a
            # single ReLU layer?
            return
        assert module not in activations, \
            f"{module} already in activations"
        # TODO [0] means first input, not all models have a single input
        inputs.append(input[0].size())
        activations.append(output.size())
        module_names.append(str(module))

    fn, hooks = hook_applyfn(store_activations, model, forward=True)
    model.apply(fn)
    with torch.no_grad():
        model(input)

    for h in hooks:
        h.remove()

    return inputs, activations, module_names


def save_layer_feature_maps_for_batch(model, input, file_prefix="",name=""):
    # activations = OrderedDict()
    feature_maps = []
    hooks = []

    def store_activations(module, input, output):
        # if isinstance(module, nn.ReLU):
        # TODO ResNet18 implementation reuses a
        # single ReLU layer?
        #     return
        # # assert module not in activations, \
        # #     f"{module} already in activations"
        # TODO [0] means first input, not all models have a single input
        # inputs.append(input[0].size())
        feature_maps.append(torch.flatten(output).detach().cpu().numpy())
        # module_names.append(str(module))

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(store_activations))

    # fn, hooks = hook_apply_partial_fn(store_activations, model, forward=True)
    # model.apply(fn)
    with torch.no_grad():
        model(input)
    for h in hooks:
        h.remove()

    for i, elem in enumerate(feature_maps):

        file_name = Path(file_prefix + "layer{}_features{}.npy".format(i,name))
        if not file_name.is_file():
            file_name.mkdir(parents=True)
        with open(file_name, "wb") as f:
            np.save(f, elem)


def load_layer_features(prefix, index,name=""):
    finished = False
    full_features = []
    while not finished:
        try:
            with open(prefix + "layer{}_features{}.npy".format(index,name), "rb") as f:
                full_features.append(np.load(f))
        except EOFError:
            finished = True

    return np.array(full_features)
#def load_model_features(prefix,number_of_layers):



