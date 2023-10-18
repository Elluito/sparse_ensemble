print("I'm about to begin the similarity comparison")
import omegaconf
from pathlib import Path
import re
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import time
from feature_maps_utils import load_layer_features
import argparse

# level 1
rf_level1_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_1_95.26.pth"
name_rf_level1_s1 = "_seed_1_rf_level_1"
rf_level1_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_1_94.93.pth"
name_rf_level1_s2 = "_seed_2_rf_level_1"

# level 2
rf_level2_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_2_94.07.pth"
name_rf_level2_s1 = "_seed_1_rf_level_2"

rf_level2_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_2_94.03.pth"
name_rf_level2_s2 = "_seed_2_rf_level_2"
# Level 3

rf_level3_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_3_92.38.pth"
name_rf_level3_s1 = "_seed_1_rf_level_3"

rf_level3_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_3_92.25.pth"
name_rf_level3_s2 = "_seed_2_rf_level_3"

# Level 4
rf_level4_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_4_90.66.pth"
name_rf_level4_s1 = "_seed_1_rf_level_4"
rf_level4_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_4_90.8.pth"
name_rf_level4_s2 = "_seed_2_rf_level_4"


def record_features_cifar10_model(architecture="resnet18", seed=1, modeltype="alternative", solution="",
                                  seed_name="_seed_1", rf_level=0):
    from feature_maps_utils import save_layer_feature_maps_for_batch

    cfg = omegaconf.DictConfig(
        {"architecture": architecture,
         "model_type": modeltype,
         "solution": solution,
         # "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         "dataset": "cifar10",
         "batch_size": 1,
         "num_workers": 0,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })
    ################################# dataset cifar10 ###########################################################################

    from alternate_models.resnet import ResNet50_rf
    from torchvision.models import resnet18, resnet50
    if cfg.dataset == "cifar10":
        if cfg.model_type == "alternative":
            resnet18_normal = ResNet50_rf(num_classes=10, rf_level=rf_level)
            resnet18_normal.load_state_dict(torch.load(cfg.solution)["net"])
        if cfg.model_type == "hub":
            resnet18_normal = resnet50()
            in_features = resnet18_normal.fc.in_features
            resnet18_normal.fc = torch.nn.Linear(in_features, 10)
            resnet18_normal.load_state_dict(torch.load(cfg.solution)["net"])

        current_directory = Path().cwd()
        data_path = "/datasets"
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
            data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"
        # data_path = "datasets" if platform.system() != "Windows" else "C:/Users\Luis Alfredo\OneDrive - " \
        #                                                                            "University of Leeds\PhD\Datasets\CIFAR10"

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)

        # cifar10_train, cifar10_val = random_split(trainset, [45000, 5000])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=False,
                                                  num_workers=cfg.num_workers)
        # val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=cfg.batch_size, shuffle=True,
        #                                          num_workers=cfg.num_workers)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers)
    if cfg.dataset == "cifar100":
        if cfg.model_type == "alternative":
            resnet18_normal = ResNet50_rf(num_classes=100, rf_level=rf_level)
            resnet18_normal.load_state_dict(torch.load(cfg.solution)["net"])
        if cfg.model_type == "hub":
            resnet18_normal = resnet50()
            in_features = resnet18_normal.fc.in_features
            resnet18_normal.fc = torch.nn.Linear(in_features, 100)
            temp_dict = torch.load(cfg.solution)["net"]
            real_dict = {}
            for k, item in temp_dict.items():
                if k.startswith('module'):
                    new_key = k.replace("module.", "")
                    real_dict[new_key] = item
            resnet18_normal.load_state_dict(real_dict)
        current_directory = Path().cwd()
        data_path = ""
        if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
            data_path = "/nobackup/sclaam/data"
        elif "luis alfredo" == current_directory.owner() or "luis alfredo" in current_directory.__str__():
            data_path = "c:/users\luis alfredo\onedrive - university of leeds\phd\datasets\cifar100"
        elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
            data_path = "datasets"

        transform_train = transforms.compose([
            transforms.randomcrop(32, padding=4),
            transforms.randomhorizontalflip(),
            transforms.totensor(),
            transforms.normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
        ])
        transform_test = transforms.compose([
            transforms.totensor(),
            transforms.normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
        ])

        trainset = torchvision.datasets.cifar100(root=data_path, train=True, download=True, transform=transform_train)

        trainloader = torch.utils.data.dataloader(trainset, batch_size=cfg.batch_size, shuffle=False,
                                                  num_workers=cfg.num_workers)
        testset = torchvision.datasets.cifar100(root=data_path, train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.dataloader(testset, batch_size=cfg.batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers)

    current_directory = Path().cwd()
    add_nobackup = ""
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        add_nobackup = "/nobackup/sclaam/"

    # prefix_custom_train = Path(
    #     "{}features/{}/{}/{}/{}/".format(add_nobackup, cfg.dataset, cfg.architecture, cfg.model_type, "train"))
    prefix_custom_test = Path(
        "{}features/{}/{}/{}/{}/".format(add_nobackup, cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    prefix_custom_test.mkdir(parents=True, exist_ok=True)
    ######################## now the pytorch implementation ############################################################
    maximun_samples = 2000
    resnet18_normal.cuda()
    o = 0
    for x, y in testloader:
        x = x.cuda()
        save_layer_feature_maps_for_batch(resnet18_normal, x, prefix_custom_test, seed_name=seed_name)
        # Path(file_prefix / "layer{}_features{}.npy".format(i, seed_name))

        print("{} batch out of {}".format(o, len(testloader)))
        if o == maximun_samples:
            break
        o += 1


def features_similarity_comparison_experiments(architecture="resnet18", modeltype1="alternative",
                                               modeltype2="alternative", name1="_seed_1", name2="_seed_2",
                                               filetype1="txt", filetype2="txt"):
    cfg = omegaconf.DictConfig(
        {"architecture": architecture,
         "model_type": modeltype1,
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
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
    prefix_modeltype1_test = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))
    cfg.model_type = modeltype2
    prefix_modeltype2_test = Path(
        "/nobackup/sclaam/features/{}/{}/{}/{}/".format(cfg.dataset, cfg.architecture, cfg.model_type, "test"))

    ##### -1 beacuse I dont have the linear layer here
    number_of_layers = int(re.findall(r"\d+", cfg.architecture)[0]) - 1

    similarity_for_networks = representation_similarity_analysis(prefix_modeltype1_test, prefix_modeltype2_test,
                                                                 number_layers=number_of_layers, name1=name1,
                                                                 name2=name2, type1=filetype1, type2=filetype2,
                                                                 use_device="cuda")
    filename = "similarity_experiments/{}_{}_V_{}_.txt".format(cfg.architecture, name1, name2)

    np.savetxt(filename, similarity_for_networks, delimiter=",")


def representation_similarity_analysis(prefix1, prefix2, number_layers, name1="", name2="", type1="txt", type2="txt",
                                       use_device="cuda"):
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
            layer_i = torch.tensor(load_layer_features(prefix1, i, name=name1, type=type1)[:1000, :])
        if use_device == "cpu":
            layer_i = load_layer_features(prefix1, i, name=name1)
        for j in range(i, number_layers):
            if use_device == "cuda":
                t0 = time.time()
                print("We are in row {} and colum {}".format(i, j))
                layer_j = torch.tensor(load_layer_features(prefix2, j, name=name2, type=type2)[:1000, :])
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
                print("Time for linear kernel: {}".format(t1 - t0))
                del layeri_cuda
                del layerj_cuda
                torch.cuda.empty_cache()

            if use_device == "cpu":
                t0 = time.time()
                print("We are in row {} and colum {}".format(i, j))
                # layer_i = load_layer_features(prefix1, i, name=name1)[:100,:]
                layer_j = load_layer_features(prefix2, j, name=name2)[:1000, :]

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

    parser.add_argument('-s', '--solution', type=str, default="", help='',
                        required=False)

    parser.add_argument('-sn1', '--seedname1', type=str, default="", help='',
                        required=True)
    parser.add_argument('-sn2', '--seedname2', type=str, default="", help='',
                        required=False)
    parser.add_argument('-rfl', '--rf_level', type=int, default=1, help='',
                        required=False)
    parser.add_argument('-e', '--experiment', type=int, default=1, help='',
                        required=False)
    parser.add_argument('-mt1', '--modeltype1', type=str, default="alternative", help='',
                        required=False)
    parser.add_argument('-mt2', '--modeltype2', type=str, default="alternative", help='',
                        required=False)
    parser.add_argument('-ft1', '--filetype1', type=str, default="txt", help='',
                        required=False)
    parser.add_argument('-ft2', '--filetype2', type=str, default="txt", help='',
                        required=False)

    #
    args = vars(parser.parse_args())
    # features_similarity_comparison_experiments(args["architecture"])

    # rf_level2_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_2_94.07.pth"
    # name_rf_level2_s1 = "_seed_1_rf_level_2"
    # rf_level2_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_2_94.03.pth"
    # name_rf_level2_s2 = "_seed_2_rf_level_2"
    # rf_level4_s1 = "trained_models/cifar10/resnet50_normal_cifar10_seed_1_rf_level_4_90.66.pth"
    # name_rf_level4_s1 = "_seed_1_rf_level_4"
    # rf_level4_s2 = "trained_models/cifar10/resnet50_normal_cifar10_seed_2_rf_level_4_90.8.pth"
    # name_rf_level4_s2 = "_seed_2_rf_level_4"
    if args["experiment"] == 1:
        record_features_cifar10_model(args["architecture"], modeltype=args["modeltype1"], solution=args["solution"],
                                      seed_name=args["seedname1"])
    #
    if args["experiment"] == 2:
        features_similarity_comparison_experiments(architecture=args["architecture"], modeltype1=args["modeltype1"],
                                                   modeltype2=args["modeltype2"], name1=args["seedname1"],
                                                   name2=args["seedname2"], filetype1=args["filetype1"],
                                                   filetype2=args["filetype2"])
