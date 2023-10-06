print("I'm about to begin the similarity comparison")
import omegaconf
from pathlib import Path
import re
import torch
import numpy as np
import time
from main import load_layer_features
import argparse


def features_similarity_comparison_experiments(architecture="resnet18"):
    cfg = omegaconf.DictConfig(
        {"architecture": architecture,
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth",
         # "solution": "trained_models/cifar10/resnet18_official_cifar10_seed_2_test_acc_88.51.pth",
         # "solution": "trained_models/cifar10/resnet18_cifar10_normal_seed_2.pth",
         # "solution": "trained_models/cifar10/resnet18_cifar10_normal_seed_3.pth",
         # explore_models_shapes()
         "dataset": "cifar10",
         "batch_size": 128,
         "num_workers": 2,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })

    prefix_custom_train = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "train"))
    prefix_custom_test = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    cfg.model_type = "hub"
    prefix_pytorch_test = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))

    ##### -1 beacuse I dont have the linear layer here
    number_of_layers = int(re.findall(r"\d+", cfg.architecture)[0]) - 1

    #########   Pytorch vs Pytorch architectures ##################################
    # similarity_for_networks = representation_similarity_analysis(prefix_pytorch_test,prefix_pytorch_test,number_layers=number_of_layers,name1="_seed_1",name2="_seed_2")
    # filename = "similarity_experiments/{}_pytorch_V_pytorch_similarity.txt".format(cfg.architecture)
    # # with open(filename,"wb") as f :
    # np.savetxt(filename,similarity_for_networks,delimiter=",")

    ######### Custom vs Custom architectures ##################################
    #

    # similarity_for_networks = representation_similarity_analysis(prefix_custom_test, prefix_custom_test,
    #                                                              number_layers=number_of_layers, name1="_seed_1",
    #                                                              name2="_seed_2", use_device="cuda")
    # filename = "similarity_experiments/{}_custom_V_custom_similarity_cuda_1000.txt".format(cfg.architecture)
    # # with open(filename,"wb") as f :
    # np.savetxt(filename, similarity_for_networks, delimiter=",")
    #
    # #########   Pytorch vs Custom architectures ##################################
    #
    similarity_for_networks = representation_similarity_analysis(prefix_pytorch_test, prefix_custom_test,
                                                                 number_layers=number_of_layers, name1="_seed_1",
                                                                 name2="_seed_1", use_device="cuda")
    filename = "similarity_experiments/{}_pytorch_V_custom_similarity_cuda_1000.txt".format(cfg.architecture)
    # with open(filename,"wb") as f :
    np.savetxt(filename, similarity_for_networks, delimiter=",")


def representation_similarity_analysis(prefix1, prefix2, number_layers, name1="", name2="", use_device="cuda"):
    from CKA_similarity.CKA import CudaCKA, CKA

    if use_device == "cuda":
        kernel = CudaCKA("cuda")
        similarity_matrix = torch.zeros((number_layers, number_layers), device=use_device)

    if use_device == "cpu":
        similarity_matrix = np.zeros((number_layers, number_layers))
        kernel = CKA()
    #### because the similiarity is a simetrical
    for i in range(number_layers):
        if use_device == "cuda":
            layer_i = torch.tensor(load_layer_features(prefix1, i, name=name1, type="npy")[:1000, :])
        if use_device == "cpu":
            layer_i = load_layer_features(prefix1, i, name=name1)
        for j in range(i, number_layers):
            if use_device == "cuda":
                t0 = time.time()
                print("We are in row {} and colum {}".format(i, j))
                layer_j = torch.tensor(load_layer_features(prefix2, j, name=name2, type="npy")[:1000, :])
                t1 = time.time()
                print("Time of loading layer: {}".format(t1 - t0))
                layeri_cuda = layer_i.cuda()
                layerj_cuda = layer_j.cuda()
                layeri_cuda = layeri_cuda - torch.mean(layeri_cuda, dtype=torch.float, dim=0)
                layerj_cuda = layerj_cuda - torch.mean(layerj_cuda, dtype=torch.float, dim=0)

                t0 = time.time()
                similarity_matrix[i, j] = kernel.linear_CKA(layeri_cuda.float(), layerj_cuda.float())
                t1 = time.time()
                t0 = time.time()
                t0 = time.time()
                print("Time for linear kernel: {}".format(t1 - t0))
                del layeri_cuda
                del layerj_cuda
                torch.cuda.empty_cache()

            if use_device == "cpu":
                t0 = time.time()
                print("We are in row {} and colum {}".format(i, j))
                # layer_i = load_layer_features(prefix1, i, name=name1)[:100,:]
                layer_j = load_layer_features(prefix2, j, name=name2)[:100, :]

                layeri_cuda = layer_i - np.mean(layer_i, dtype=np.float, axis=0)
                layerj_cuda = layer_j - np.mean(layer_j, dtype=np.float, axis=0)

                similarity_matrix[i, j] = kernel.linear_CKA(layeri_cuda, layerj_cuda)
                t1 = time.time()
                print("Time of loading + linear kernel: {}".format(t1 - t0))
                del layeri_cuda
                del layeri_cuda
                del layerj_cuda

    # network1 =
    if use_device == "cuda":
        simetric_similarity = similarity_matrix.add(similarity_matrix.T)
        simetric_similarity[range(number_layers), range(number_layers)] *= 1 / 2
        return simetric_similarity.detach().cpu().numpy()
    if use_device == "cpu":
        simetric_similarity = similarity_matrix + np.transpose(similarity_matrix)
        simetric_similarity[range(number_layers), range(number_layers)] *= 1 / 2
        return simetric_similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Similarity experiments')
    parser.add_argument('-arch', '--architecture', type=str, default="resnet18", help='Architecture for analysis',
                        required=True)
    args = vars(parser.parse_args())
    features_similarity_comparison_experiments(args["architecture"])
