import torch
from alternate_models.resnet import ResNet50_rf
import pandas as pd
from main import prune_function, remove_reparametrization, get_layer_dict, get_datasets, count_parameters
from sparse_ensemble_utils import test
import omegaconf

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
files_names = [name_rf_level1_s1, name_rf_level1_s2, name_rf_level2_s1, name_rf_level2_s2, name_rf_level3_s1,
               name_rf_level3_s2, name_rf_level4_s1, name_rf_level4_s2]
files = [rf_level1_s1, rf_level1_s2, rf_level2_s1, rf_level2_s2, rf_level3_s1, rf_level3_s2, rf_level4_s1, rf_level4_s2]


def main():
    cfg = omegaconf.DictConfig(
        {"architecture": "resnet50",
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": "cifar10",
         "batch_size": 128,
         "num_workers": 2,
         "amount": 0.9,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": ["conv1", "linear"]

         })
    train, val, testloader = get_datasets(cfg)

    dense_accuracy_list = []
    pruned_accuracy_list = []

    for i in range(len(files)):
        state_dict_raw = torch.load(files[i])
        dense_accuracy_list.append(state_dict_raw["acc"])
        net = ResNet50_rf(num_classes=10)
        net.load_state_dict(state_dict_raw["net"])
        prune_function(net, cfg)
        remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)
        pruned_accuracy = test(net, use_cuda=True, testloader=testloader, verbose=0)
        pruned_accuracy_list.append(pruned_accuracy)
        weight_names, weights = zip(*get_layer_dict(net))
        total_params = count_parameters(net)
        zero_number = lambda w: (torch.count_nonzero(w == 0) / total_params).cpu().numpy()
        pruning_rates_per_layer = list(map(zero_number, weights))
        df2 = pd.DataFrame({"layer_names": weight_names, "pr": pruning_rates_per_layer})
        df2.to_csv("{}_pruning_rates.csv".format(files_names[i]), index=False)
    df = pd.DataFrame({"Name": files_names,
                       "Dense Accuracy": dense_accuracy_list,
                       "Pruned Accuracy": pruned_accuracy_list,
                       })
    df.to_csv("RF_summary.csv", index=False)
if __name__ == '__main__':
    main()