import argparse

from CKA_similarity.CKA import CudaCKA, CKA
from sparse_ensemble_utils import apply_mask

import torch

import numpy as np

from torchvision import transforms as trs

import torch.nn as nn

from pathlib import Path

from main import get_datasets, prune_with_rate, get_layer_dict, remove_reparametrization, test

from alternate_models import ResNet18_rf, ResNet24_rf, ResNet50_rf, VGG_RF, small_VGG_RF, small_ResNetRF

from torchvision.models import resnet18, resnet50

from npy_append_array import NpyAppendArray

import omegaconf
import copy

if torch.cuda.is_available():
    use_device = "cuda"
else:
    use_device = "cpu"


def build_model(args):
    net = None
    if args.model == "resnet18":

        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet18_rf(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
    if args.model == "resnet50":

        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level, multiplier=args.width)

        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)

    if args.model == "resnet24":

        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
        if args.type == "pytorch" and args.dataset == "cifar100":
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 100)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
    if args.model == "vgg19":

        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, rf_level=args.RF_level)

        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, rf_level=args.RF_level)
    if args.model == "resnet_small":
        if args.type == "normal" and args.dataset == "cifar10":
            net = small_ResNetRF(num_classes=10, RF_level=args.RF_level, multiplier=args.width)

        if args.type == "normal" and args.dataset == "cifar100":
            net = small_ResNetRF(num_classes=100, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_ResNetRF(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg_small":

        if args.type == "normal" and args.dataset == "cifar10":
            net = small_VGG_RF("small_vgg", num_classes=10, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "cifar100":
            net = small_VGG_RF("small_vgg", num_classes=100, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_VGG_RF("small_vgg", num_classes=200, RF_level=args.RF_level)

    if not net:
        raise Exception(
            "Configuration not valid model {}, type {}, dataset {}".format(args.model, agrs.type, args.dataset))


def load_solution(model, solution):
    temp_dict = torch.load(solution)["net"]
    real_dict = {}
    for k, item in temp_dict.items():
        if k.startswith('module'):
            new_key = k.replace("module.", "")
            real_dict[new_key] = item
    model.load_state_dict(real_dict)


def save_layer_feature_maps_for_batch_channels(model, input, file_prefix="", seed_name=""):
    model.eval()

    # activations = OrderedDict()

    feature_maps = []

    hooks = []

    if use_device == "cuda":
        kernel = CudaCKA("cuda")

    if use_device == "cpu":
        kernel = CKA()

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

        batch_size, out_channels, y_dim, x_dim = output.shape
        similarity_matrix = torch.zeros((out_channels, out_channels), device=use_device)
        reshaped_output = torch.reshape(output, (batch_size, out_channels, y_dim * x_dim))

        for i in range(out_channels):

            for j in range(i, out_channels):
                channel_i = reshaped_output[:, i, :]
                channel_j = reshaped_output[:, j, :]
                similarity_between_channels = kernel.linear_CKA(channel_i, channel_j)
                similarity_matrix[i, j] = similarity_between_channels.item()

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

    for i, elem in enumerate(feature_maps):
        #
        #
        file_name = Path(file_prefix / "sim_matrix_layer_{}_{}.npy".format(i, seed_name))
        # if not file_name.is_file():
        #     file_name.mkdir(parents=True)
        # suma = np.sum(elem == 0)
        # n = len(elem)
        # print(n)
        # print("Current layer: {}".format(i))
        # print("{} out of {} elements are 0".format(suma, n))

        with NpyAppendArray(file_name) as npaa:
            npaa.append(elem)

        # with open(file_name, "a+") as f:

        #     np.savetxt(f, elem.reshape(1, -1), delimiter=",")

        print("Udated file {}".format(file_name))

    return feature_maps


def load_sim_matrix(prefix, index, name="", type="txt"):
    # reading_string = "r" if type == "txt" else "rb"
    with open(prefix / "sim_matrix_layer_{}_{}.npy".format(index, name), "rb") as f:
        # file_name = Path(prefix / "sim_matrix_layer_{}_{}.npy".format(i, seed_name))
        # full_features.append(np.load(f))

        # if type == "txt":
        #     features = np.loadtxt(f, delimiter=",")
        # if type == "npy":

        features = np.load(f)
    return features


def calculate_per_layer_filter_rate(weights_dict, lamp_pruning_rates_per_layer):
    output_dict = {}
    for i, (name, pr_rate) in enumerate(lamp_pruning_rates_per_layer.items()):
        current_w = weights_dict[i]
        input_channel, output_channel, x, y = current_w.shape
        kernel_size = x
        number_weights_in_kernel = x * y

        number_weights_in_output_channel = number_weights_in_kernel * input_channel

        number_weights_from_pr = pr_rate * current_w.nelement()

        number_of_filters_to_prune = number_weights_from_pr // number_weights_in_output_channel

        # number_of_kernels = current_w.nelement() / number_weights_in_kernel

        # pruning_rate_for_filters = nominator / number_of_kernels

        output_dict[name] = number_of_filters_to_prune

    return output_dict


def prune_with_similarity(args):
    type = args.modeltype1
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": args.architecture,
         "type": type,
         "solution": args.solution,
         # "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         "dataset": "cifar10",
         "batch_size": args.batch_size,
         "num_workers": 0,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": exclude_layers

         })
    train_loader, val_loader, test_loader = get_datasets(args)
    model = build_model(args)
    load_solution(model, args.solution)
    dummy_net = copy.deepcopy(model)

    pr_per_layer_erk = prune_with_rate(dummy_net, cfg.amount, pruner="erk", type="layer-wise",
                                       return_pr_per_layer=True)

    weights_dict = dict(get_layer_dict(model))

    number_of_filters_to_prune = calculate_per_layer_filter_rate(weights_dict, pr_per_layer_erk)

    # pruned_accuracy = test(net, use_cuda=True, testloader=testloader, verbose=0)

    prefix_custom_train = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/{}/".format(args.dataset, args.architecture, args.model_type, "train",
                                                           "similarity_matrix"))

    prefix_modeltype1_test = Path(

        "/nobackup/sclaam/features/{}/{}/{}/{}/{}/".format(args.dataset, args.architecture, args.model_type, "test",
                                                           "similarity_matrix"))

    # cfg.model_type = modeltype2

    prefix_modeltype1_val = Path(

        "/nobackup/sclaam/features/{}/{}/{}/{}/{}/".format(args.dataset, args.architecture, args.type, "test",
                                                           "similarity_matrix"))

    prefix_modeltype1_val.mkdir(parents=True, exist_ok=True)
    mask_for_weigths = {}

    for i, (name, weight) in enumerate(weights_dict.items()):
        # save_layer_feature_maps_for_batch_channels(model, x, prefix_modeltype1_val, args.seedname1)

        # This is a number_of_batches X number_of_output_channels  X number_of_output_channels  similarity matrix
        val_loader_sim_matrices = load_sim_matrix(prefix_modeltype1_val, i, args.seedname1, type="npy")
        # Lower triangular is all 0

        mean_sim_matrix = np.mean(val_loader_sim_matrices, axis=0)

        vector_mean_sim_matrix = np.ravel(mean_sim_matrix)

        # This gives smallest values firsts

        most_similar_channels = np.argsort(vector_mean_sim_matrix)

        vector_copy = vector_mean_sim_matrix.copy()

        current_number_of_filters_to_prune = number_of_filters_to_prune[name]

        # pruning_index = int((mean_sim_matrix.shape[0]) * current_pr)
        # The index on the vector corresponding to the most similar channels are set to -100

        vector_copy[most_similar_channels[-current_number_of_filters_to_prune:]] = -100

        matrix = np.reshape(vector_copy, mean_sim_matrix.shape)

        mask = torch.ones_like(weight)
        # Output Channel i
        for ii in range(mean_sim_matrix.shape[0]):
            # Output Channel j
            for j in range(ii, mean_sim_matrix.shape[0]):

                if matrix[ii, j] == -100:
                    # Eliminate all weigths for output channel i
                    mask[:, ii, :, :] = 0

        mask_for_weigths[name] = mask

    new_model = copy.deepcopy(model)
    apply_mask(new_model, mask_for_weigths)

    test_accuracy = test(new_model, True, test_loader)

    print("Test accuracy with whole model pruning of {} is {}".format(cfg.amount, test_accuracy))


def main(args):
    type = args.modeltype1

    cfg = omegaconf.DictConfig(
        {"architecture": args.architecture,
         "type": type,
         "solution": args.solution,
         # "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         "dataset": "cifar10",
         "batch_size": args.batch_size,
         "num_workers": 0,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]
         })
    train_loader, val_loader, test_loader = get_datasets(args)
    model = build_model(args)
    load_solution(model, args.solution)

    prefix_custom_train = Path(

        "/nobackup/sclaam/features/{}/{}/{}/{}/{}/".format(args.dataset, args.architecture, args.model_type, "train",
                                                           "similarity_matrix"))

    prefix_modeltype1_test = Path(

        "/nobackup/sclaam/features/{}/{}/{}/{}/{}/".format(args.dataset, args.architecture, args.model_type, "test",
                                                           "similarity_matrix"))

    # cfg.model_type = modeltype2

    prefix_modeltype1_val = Path(

        "/nobackup/sclaam/features/{}/{}/{}/{}/{}/".format(args.dataset, args.architecture, args.type, "test",
                                                           "similarity_matrix"))

    prefix_modeltype1_val.mkdir(parents=True, exist_ok=True)
    for x, y in val_loader:
        save_layer_feature_maps_for_batch_channels(model, x, prefix_modeltype1_val, args.seedname1)


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
    # parser.add_argument('--RF_level', default=4, type=int, help='Receptive field level')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', default="cifar10", type=str,
                        help='Dataset to use [cifar10,cifar100,tiny_imagenet]')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size for training')
