import numpy as np
import omegaconf
from pathlib import Path
import re
from feature_maps_utils import load_layer_features
import time
import torch
from CKA_similarity.CKA import CKA, CudaCKA

architecture = "resnet50"
kernel = CudaCKA("cuda")
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

name1 = "_seed_1"
name2 = "_seed_2"


def convert_to_npy(prefix1, name1):
    indexes = range(49)
    for i in indexes:
        t0 = time.time()
        thing = torch.tensor(load_layer_features(prefix1, i, name=name1, type="npy")[1000:,:], device="cuda")
        t1 = time.time()
        print("Time loading layer {} npy: {}".format(i, t1 - t0))
        t0 = time.time()
        self_similarity = kernel.linear_CKA(thing.float(), thing.float())
        t1 = time.time()
        print("Self similarity {} calculated in {}".format(self_similarity, t1 - t0))

        # file_name = Path(prefix1 / "layer{}_features{}.npy".format(i, name1))
        # np.save(file_name, thing)


convert_to_npy(prefix_custom_test, name1)
# convert_to_npy(prefix_custom_test, name2)
