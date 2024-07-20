import functools
from collections import OrderedDict, defaultdict
from typing import List, Any
import numpy as np
from pathlib import Path
import torch
from torch import nn
from npy_append_array import NpyAppendArray

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


def save_layer_feature_maps_for_batch(model, input, file_prefix="", seed_name=""):
    model.eval()
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
        # print("input")
        # print(input)
        # print("OUTPUT")
        # print(output)
        # return
        feature_maps.append(torch.flatten(output).detach().cpu().numpy())
        # module_names.append(str(module))

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(store_activations))
            # break
    #
    # fn, hooks = hook_apply_partial_fn(store_activations, model, forward=True)
    # model.apply(fn)
    with torch.no_grad():
        model(input)
    for h in hooks:
        h.remove()

    # Path(file_prefix).mkdir(parents=True, exist_ok=True)


    for i, elem in enumerate(feature_maps):

        file_name = Path(file_prefix / "layer{}_features{}.npy".format(i, seed_name))
        # if not file_name.is_file():
        #     file_name.mkdir(parents=True)
        # suma = np.sum(elem == 0)
        # n = len(elem)
        # print(n)
        # print("Current layer: {}".format(i))
        # print("{} out of {} elements are 0".format(suma, n))

        with NpyAppendArray(file_name) as npaa:
            npaa.append(elem.reshape(1, -1))
        # with open(file_name, "a+") as f:
        #     np.savetxt(f, elem.reshape(1, -1), delimiter=",")
        print("Udated file {}".format(file_name))
    return feature_maps



def load_layer_features(prefix, index, name="", type="txt"):
    finished = False
    features = None
    counter = 0
    reading_string = "r" if type == "txt" else "rb"
    with open(prefix / "layer{}_features{}.{}".format(index, name, type), reading_string) as f:
        # full_features.append(np.load(f))
        if type == "txt":
            features = np.loadtxt(f, delimiter=",")
        if type == "npy":
            features = np.load(f)
    # while not finished:
    #     try:
    #             suma = np.sum(feature == 0)
    #             n = len(feature)
    #             print("Current sample: {}".format(counter))
    #             print("{} out of {} elements are 0".format(suma, n))
    #             counter += 1
    #             # if counter == 9:
    #             # return full_features
    #     except EOFError:
    #         finished = True
    return features
# def load_model_features(prefix,number_of_layers):
