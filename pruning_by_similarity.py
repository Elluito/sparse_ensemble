import argparse
from CKA_similarity.CKA import CudaCKA, CKA
import torch
import numpy as np
from torchvision import transforms as trs
import torch.nn as nn
from pathlib import Path
from main import get_datasets, get_model
from npy_append_array import NpyAppendArray

if torch.cuda.is_available():
    use_device = "cuda"
else:
    use_device = "cpu"


def save_layer_feature_maps_for_batch_channels(model, input, file_prefix="", seed_name=""):
    model.eval()
    # activations = OrderedDict()
    feature_maps = []
    hooks = []

    def store_similarity_matrix(module, input, output):
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
        if use_device == "cuda":
            kernel = CudaCKA("cuda")
        if use_device == "cpu":
            kernel = CKA()

        batch_size, out_channels, y_dim, x_dim = output.shape
        similarity_matrix = torch.zeros((out_channels, out_channels), device=use_device)
        reshaped_output = torch.reshape(output, (batch_size, out_channels, y_dim * x_dim))
        for i in range(out_channels):
            for j in range(i, out_channels):
                channel_i = reshaped_output[:, i, :]
                channel_j = reshaped_output[:, j, :]
                similarity_between_channels = kernel.linear_CKA(channel_i, channel_j)
                similarity_matrix[i, j] = similarity_between_channels
        feature_maps.append(similarity_matrix)
        # module_names.append(str(module))

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(store_similarity_matrix))
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
        file_name = Path(file_prefix / "sim_layer_{}_{}.npy".format(i, seed_name))
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


def main(args):
    if use_device == "cuda":
        kernel = CudaCKA("cuda")
        # similarity_matrix = torch.zeros((number_layers, number_layers), device=use_device)

    if use_device == "cpu":
        # similarity_matrix = np.zeros((number_layers, number_layers))
        kernel = CKA()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Similarity pruning experiments')
    parser.add_argument('-arch', '--architecture', type=str, default="resnet18", help='Architecture for analysis',
                        required=True)
    parser.add_argument('-dt', '--dataset', type=str, default="CIFA10", help='Dataset for analysis',
                        required=True)
    parser.add_argument('-s', '--solution', type=str, default="", help='',
                        required=False)

    parser.add_argument('-sn1', '--seedname1', type=str, default="", help='',
                        required=True)
    parser.add_argument('-sn2', '--seedname2', type=str, default="", help='',
                        required=False)
    parser.add_argument('-rfl', '--rf_level', type=int, default=1, help='',
                        required=False)
    parser.add_argument('-li', '--layer_index', type=int, default=0, help='',
                        required=False)
    parser.add_argument('-e', '--experiment', type=int, default=1, help='',
                        required=False)
    parser.add_argument('-mt1', '--modeltype1', type=str, default="alternative", help='',
                        required=False)
    parser.add_argument('-mt2', '--modeltype2', type=str, default="alternative", help='',
                        required=False)
    parser.add_argument('-ft1', '--filetype1', type=str, default="npy", help='',
                        required=False)
    parser.add_argument('-ft2', '--filetype2', type=str, default="npy", help='',
                        required=False)
    parser.add_argument('-t', '--train', type=int, default=1, help='',
                        required=False)
