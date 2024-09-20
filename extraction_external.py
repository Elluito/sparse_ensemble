# This file was taken from  https://github.com/delve-team/phd-lab/blob/master/phd_lab/experiments/utils/extraction.py#L109
from torch.nn import Module
import torch
import omegaconf
import os
from os import makedirs
from os.path import exists, join
from torch.nn import Conv2d, Linear, LSTM
from torch.utils.data import DataLoader
from typing import Union, Dict, List
from shutil import rmtree
from itertools import product
import numpy as np
from time import time
import pickle
import argparse
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import random_split
import torch.nn as nn
from alternate_models import *
from typing import Optional

if os.name == 'nt':  # running on windows:
    import win32file

    win32file._setmaxstdio(2048)


class Extract:
    latent_representation_logs: str = './latent_datasets/'
    downsampling: Optional[int] = 4
    save_feature_map_positions_individually: bool = False

    # def __call__(self, trainer: Trainer):
    def __call__(self, model, model_name, dataset_name, data_resolution, train_loader, test_loader, device, save_path,
                 RF_level, name=None):
        # trainer._tracker.stop()
        # model = trainer.model if not isinstance(trainer.model, DataParallel) else trainer.model.module
        print('Initializing logger')
        save_name = '{}_{}_{}_{}'.format(
            # model.module.name if isinstance(model, DataParallel) else model.name,
            model_name,
            # trainer.data_bundle.dataset_name,
            dataset_name,
            # trainer.data_bundle.output_resolution
            data_resolution,
            RF_level
        )
        if name:
            save_name = '{}_{}_{}_{}'.format(
                # model.module.name if isinstance(model, DataParallel) else model.name,
                model_name,
                # trainer.data_bundle.dataset_name,
                dataset_name,
                # trainer.data_bundle.output_resolution
                data_resolution,
                RF_level,
                name
            )

        logger = LatentRepresentationCollector(
            model,
            savepath=os.path.join(
                self.latent_representation_logs,
                save_name
            ),
            downsampling=self.downsampling,
            save_per_position=self.save_feature_map_positions_individually,
            overwrite=True

        )
        print('Extracting training')
        model.train()
        # extract_from_dataset(logger, True, model, trainer.data_bundle.train_dataset, trainer.device)
        extract_from_dataset(logger, True, model, train_loader, device)
        # logger.save(os.path.dirname(trainer._save_path))
        logger.save(os.path.dirname(save_path))
        print('Extracting test')
        model.eval()
        # extract_from_dataset(logger, False, model, trainer.data_bundle.test_dataset, trainer.device)
        extract_from_dataset(logger, False, model, test_loader, device)
        # logger.save(os.path.dirname(trainer._save_path))
        logger.save(os.path.dirname(save_path))


class LatentRepresentationCollector:
    """This Object collects the latent representation from all layers.
    """

    def __init__(self, model: Module,
                 savepath: str,
                 save_instantly: bool = True,
                 downsampling: int = None,
                 save_per_position: bool = False,
                 overwrite: bool = False):
        """

        Args:
            model:              this is a pyTorch-Module
            savepath:           the filepath points to a folder where latent representations will be stored.
                                For storage a subfolder will be created.
            save_instantly:     if true, the data will be saved incrementally with a save checkpoint at each batch.
            downsampling:       downsample the latent representation to a height and width equal to the downsampling value.
            save_per_position:  saves a dataset per layer per position of the feature map instead of saving the feature maps downsamples as a whole.
        """
        self.savepath = savepath
        self.downsampling = downsampling
        self.save_per_position = save_per_position
        self.overwrite = overwrite
        self.pre_exists = False
        if exists(savepath) and overwrite:
            print('Found previous extraction in folder, removing it...')
            rmtree(savepath)
        if not exists(savepath):
            makedirs(self.savepath)
        else:
            self.pre_exists = True
        self.layers = self.get_layers_recursive(model)
        for name, layer in self.layers.items():
            if isinstance(layer, Conv2d) or isinstance(layer, Linear) \
                    or isinstance(layer, LSTM):
                self._register_hooks(layer=layer,
                                     layer_name=name,
                                     interval=1)
        self.save_instantly = save_instantly

        self.logs = {
            'train': {},
            'eval': {}
        }
        self.record = True

    def _record_stat(self, activations_batch: torch.Tensor, layer: Module, training_state: str):
        """This function is called in the forward-hook to all convolutional and linear layers.

        Args:
            activations_batch:  the batch of data
            layer:              the module object, the latent representations are recorded.
            training_state:     state of training, may be either "eval" or "train"

        Returns:
            Returns nothing, this hook is side-effect free
        """
        if activations_batch.dim() == 4:  # conv layer (B x C x H x W)
            if self.downsampling is not None:
                # activations_batch = torch.nn.functional.interpolate(activations_batch, self.downsampling, mode="nearest")
                activations_batch = torch.nn.functional.adaptive_avg_pool2d(activations_batch,
                                                                            (self.downsampling, self.downsampling))
            if not self.save_per_position:
                activations_batch = activations_batch.view(activations_batch.size(0), -1)
        batch = activations_batch.cpu().numpy()
        if not self.save_instantly:
            if layer.name not in self.logs[training_state]:
                self.logs[training_state][layer.name] = batch
            else:
                self.logs[training_state][layer.name] = np.vstack((self.logs[training_state][layer.name], batch))
        elif self.save_per_position and len(batch.shape) == 4:
            for (i, j) in product(range(batch.shape[2]), range(batch.shape[3])):
                position = batch[:, :, i, j]
                saveable = position.squeeze()
                savepath = self.savepath + '/' + training_state + '-' + layer.name + f'-({i},{j})' + '.p'
                if not exists(savepath):
                    if layer.name not in self.logs[training_state]:
                        self.logs[training_state][layer.name] = {}
                    self.logs[training_state][layer.name][(i, j)] = savepath

                with open(self.logs[training_state][layer.name][(i, j)], 'ab') as fp:
                    pickle.dump(saveable, file=fp)

        else:
            savepath = self.savepath + '/' + training_state + '-' + layer.name + '.p'
            if not exists(savepath):
                self.logs[training_state][layer.name] = open(savepath, 'wb')
            pickle.dump(batch, file=self.logs[training_state][layer.name])

    def _register_hooks(self, layer: Module, layer_name: str, interval: int) -> None:
        """Register a forward hook on a given layer.

        Args:
            layer:          the module.
            layer_name:     name of the layer.
            interval:       unused variable, needed for compatibility.
        """
        layer.name = layer_name

        def record_layer_history(layer: torch.nn.Module, input, output):
            """Hook to register in `layer` module."""

            if not self.record:
                return

            # Increment step counter
            # layer.forward_iter += 1

            training_state = 'train' if layer.training else 'eval'
            activations_batch = output.data
            self._record_stat(activations_batch, layer, training_state)

        layer.register_forward_hook(record_layer_history)

    def get_layer_from_submodule(self, submodule: torch.nn.Module,
                                 layers: dict, name_prefix: str = '') -> Dict[str, Module]:
        """Finds all linear and convolutional layers in a network structure.

        The algorithm is recursive.

        Args:
            submodule:      the current submodule.
            layers:         the dictionary containing all layers found so far.
            name_prefix:    the prefix of the layers name. The prefix resembled the position in
                            the networks structure.

        Returns:
            the layers stored in a dictionary.
        """
        if len(submodule._modules) > 0:
            for idx, (name, subsubmodule) in \
                    enumerate(submodule._modules.items()):
                new_prefix = name if name_prefix == '' else name_prefix + \
                                                            '-' + name
                self.get_layer_from_submodule(subsubmodule, layers, new_prefix)
            return layers
        else:
            layer_name = name_prefix
            layer_type = layer_name
            if not isinstance(submodule, Conv2d) and not \
                    isinstance(submodule, Linear) and not \
                    isinstance(submodule, LSTM):
                print(f"Skipping {layer_type}")
                return layers
            layers[layer_name] = submodule
            print('added layer {}'.format(layer_name))
            return layers

    def get_layers_recursive(self, modules: Union[List[torch.nn.Module], torch.nn.Module]) -> Dict[str, Module]:
        """Recursive search algorithm for finding convolutional an linear layers

        Args:
            modules: maybe a single (sub)-module or a List of modules

        Returns:
            a dictionary that maps layer names to modules
        """
        layers = {}
        if not isinstance(modules, list) and not hasattr(
                modules, 'out_features'):
            # is a model with layers
            # check if submodule
            submodules = modules._modules  # OrderedDict
            layers = self.get_layer_from_submodule(modules, layers, '')
        else:
            for module in modules:
                layers = self.get_layer_from_submodule(module, layers, '')
        return layers

    def save(self, model_log_path) -> None:
        """Saving the models latent representations.

        Args:
            model_log_path:     the path that logs the model.

        """
        with open(join(self.savepath, "model_pointer.txt"), "w+") as fp:
            fp.write(model_log_path)
        if not exists(self.savepath):
            makedirs(self.savepath)
        for mode, logs in self.logs.items():
            for layer_name, data in self.logs[mode].items():
                if isinstance(data, str):
                    continue
                if isinstance(data, np.ndarray):
                    with open(self.savepath + '/' + mode + '-' + layer_name + '.p', 'wb') as p:
                        pickle.dump(data, p)
                else:
                    if isinstance(data, dict):
                        for _, fp in data.items():
                            if hasattr(fp, "close"):
                                fp.close()
                    else:
                        if hasattr(fp, "close"):
                            data.close()


def main(args):
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear", "fc"]

    print("Normal data loaders loaded!!!!")

    cifar10_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    cifar100_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    stats_to_use = cifar10_stats if args.dataset == "cifar10" else cifar100_stats
    # Data
    print('==> Preparing data..')
    # current_directory = args.folder
    data_path = args.folder
    # if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
    #     data_path = "/nobackup/sclaam/data"
    # elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
    #     data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
    # elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
    #     data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
    # elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
    #     data_path = "/home/luisaam/Documents/PhD/data/"
    print(data_path)
    batch_size = args.batch_size
    # if "32" in args.name:
    #     batch_size = 32
    # if "64" in args.name:
    #     batch_size = 64

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats_to_use),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform_train)

        cifar10_train, cifar10_val = random_split(trainset, [len(trainset) - 5000, 5000])

        trainloader = torch.utils.data.DataLoader(
            cifar10_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        valloader = torch.utils.data.DataLoader(
            cifar10_val, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    if args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform_train)

        cifar10_train, cifar10_val = random_split(trainset, [len(trainset) - 5000, 5000])

        trainloader = torch.utils.data.DataLoader(
            cifar10_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        valloader = torch.utils.data.DataLoader(
            cifar10_val, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    if args.dataset == "tiny_imagenet":
        from test_imagenet import load_tiny_imagenet
        trainloader, valloader, testloader = load_tiny_imagenet(
            {"traindir": data_path + "/tiny_imagenet_200/train", "valdir": data_path + "/tiny_imagenet_200/val",
             "num_workers": args.num_workers, "batch_size": batch_size})
    if args.dataset == "small_imagenet":
        if args.ffcv:
            from ffcv_loaders import make_ffcv_small_imagenet_dataloaders
            trainloader, valloader, testloader = make_ffcv_small_imagenet_dataloaders(args.ffcv_train,
                                                                                      args.ffcv_val,
                                                                                      batch_size, args.num_workers,
                                                                                      valsize=5000,
                                                                                      testsize=10000,
                                                                                      shuffle_val=True,
                                                                                      shuffle_test=False, )
        else:

            from test_imagenet import load_small_imagenet
            trainloader, valloader, testloader = load_small_imagenet(
                {"traindir": data_path + "/small_imagenet/train", "valdir": data_path + "/small_imagenet/val",
                 "num_workers": args.num_workers, "batch_size": batch_size, "resolution": args.input_resolution},
                val_size=5000, test_size=10000, shuffle_val=True, shuffle_test=False)

    from torchvision.models import resnet18, resnet50

    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level)

        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg19":
        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet24":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
    if args.model == "resnet_small":
        if args.type == "normal" and args.dataset == "cifar10":
            net = small_ResNet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = small_ResNet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            # raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            # raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)

    ###########################################################################
    if args.solution:

        temp_dict = torch.load(args.solution, map_location=torch.device('cpu'))["net"]
        if args.type == "normal" and args.RF_level != 0:
            net.load_state_dict(temp_dict, strict=False)
            print("Loaded solution!")
        else:
            real_dict = {}
            for k, item in temp_dict.items():
                if k.startswith('module'):
                    new_key = k.replace("module.", "")
                    real_dict[new_key] = item
            net.load_state_dict(real_dict, strict=False)
            print("Loaded solution!")
    extractor = Extract()
    net.to(args.device)
    extractor(net, args.model, args.dataset, args.input_resolution, trainloader, testloader, args.device,
              args.save_path, args.RF_level,args.name)


def extract_from_dataset(logger: LatentRepresentationCollector,
                         train: bool, model: Module, dataset: DataLoader, device: str) -> None:
    """Extract latent representations from a given classification dataset.

    Args:
        logger:     The logger that collects the latent representations.
        train:      Marks the subset as training or evalutation dataset
        model:      The model from which the latent representations need to be collected.
        dataset:    The dataset, may be a torch data-loader
        device:     The device the model is deployed on, maybe any torch compatible key.
    """
    if logger.pre_exists:
        print("Found existing latent representations and overwrite is disabled")
        return
    mode = 'train' if train else 'eval'
    correct, total = 0, 0
    old_time = time()
    with torch.no_grad():
        for batch, data in enumerate(dataset):
            if batch % 1 == 0 and batch != 0:
                print(batch, 'of', len(dataset), 'processing time', time() - old_time, ' acc:', correct / total)
                old_time = time()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
            if 'labels' not in logger.logs[mode]:
                logger.logs[mode]['labels'] = labels.cpu().numpy()
            else:
                logger.logs[mode]['labels'] = np.hstack((logger.logs[mode]['labels'], labels.cpu().numpy()))
    print('accuracy:', correct / total)


def run_local_test():
    cfg = omegaconf.DictConfig({
        # "solution": "/home/luisaam/checkpoints/resnet_small_normal_small_imagenet_seed.8_rf_level_5_recording_200_test_acc_62.13.pth",
        "solution": "/home/luisaam/checkpoints/vgg19_normal_cifar10_1723720946.9104598_rf_level_1_recording_200_no_ffcv_test_acc_93.77.pth",
        # "solution": "/home/luisaam/checkpoints/resnet50_normal_cifar10_1723722961.8540442_rf_level_2_recording_200_no_ffcv_test_acc_94.24.pth",
        "modeltype1": "normal",
        "seedname1": "_seed_8",
        "RF_level": 2,
        "epochs": 1,
        "ffcv": 0,
        "ffcv_val": "",
        "ffcv_train": "",
        "batch_size": 64,
        "model": "resnet50",
        "dataset": "cifar10",
        "num_workers": 0,
        "input_resolution": 32,
        "width": 1,
        "name": "no_name",
        "save_path": "./logs/",
        "folder": "/home/luisaam/Documents/PhD/data/",
        "lr": 0.1,
        "device": "cpu",
        "type": "normal",
        "resume": False,
        "eval_size": 5000,

    })

    main(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='folder', type=str, default="./latent_datasets", help='data folder')
    parser.add_argument('-m', dest='model', type=str, default="vgg19", help='Model architecture')
    parser.add_argument('-t', dest='type', type=str, default="normal", help='Type of model')
    parser.add_argument('-s', dest='solution', type=str, default=None, help='Solution to use')
    parser.add_argument('-d', dest='dataset', type=str, default=None, help='Dataset to use')
    parser.add_argument('--name', dest='name', type=str, default=None, help='Name of the files')
    parser.add_argument('--RF_level', dest='RF_level', type=int, default=2, help='Receptive field level')
    parser.add_argument('--device', dest='device', type=str, default="cuda:0", help='Device to use')
    parser.add_argument('--input_resolution', dest='input_resolution', type=int, default=32, help='Input resolution')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('--width', default=1, type=int, help='Width of the Network')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--save_path', default="./probes_logs/", type=str, help='Save path of logs')
    args = parser.parse_args()

    main(args)

    # run_local_test()
