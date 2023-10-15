import numpy as np
import argparse
import omegaconf
from pathlib import Path
import re
from feature_maps_utils import load_layer_features
import time
import torch
from CKA_similarity.CKA import CKA, CudaCKA
def convert_to_npy(prefix1, name1,indexes=range(49)):
    for i in indexes:
        t0 = time.time()
        thing = load_layer_features(prefix1, i, name=name1,type="txt")
        # thing = load_layer_features(prefix1, i, name=name1, type="npy")
        # print("sizr of thing {}".format(thing.size()))
        t1 = time.time()
        print("Time loading layer {} npy: {}".format(i, t1 - t0))
        # t0 = time.time()
        # self_similarity = kernel.linear_CKA(thing.float(), thing2.float())
        # t1 = time.time()
        # print("Self similarity {} calculated in {}".format(self_similarity, t1 - t0))

        file_name = Path(prefix1 / "layer{}_features{}.npy".format(i, name1))
        np.save(file_name, thing)

def main(args):
    architecture = args.architecture
    modeltype=args.modeltype
    cfg = omegaconf.DictConfig(
        {"architecture": architecture,
         # "model_type": "alternative",
         "model_type": modeltype,
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

    # prefix_custom_train = Path(
    #     "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "train"))
    # prefix_custom_test = Path(
    #     "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    # cfg.model_type = "hub"
    prefix_to_use_test = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))

    ##### -1 beacuse I dont have the linear layer here
    number_of_layers = int(re.findall(r"\d+", cfg.architecture)[0]) - 1

    name1 = args.seedname1



    convert_to_npy(prefix_to_use_test, name1)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Similarity experiments')
    parser.add_argument('-arch', '--architecture', type=str, default="resnet18", help='Architecture for analysis',
                        required=True)

    parser.add_argument('-s', '--solution', type=str, default="", help='',
                        required=False)
    parser.add_argument('-m', '--modeltype', type=str, default="alternative", help='',
                        required=False)
    parser.add_argument('-d', '--dataset', type=str, default="cifar10", help='',
                        required=False)

    parser.add_argument('-sn1', '--seedname1', type=str, default="", help='',
                        required=False)
    # parser.add_argument('-sn2', '--seedname2', type=str, default="", help='',
    #                     required=False)

    args = vars(parser.parse_args())
    main(args=args)
