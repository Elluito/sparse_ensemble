import argparse
import glob
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
import time
from main import get_model
from sparse_ensemble_utils import sparsity, test_with_accelerator
import omegaconf
from torch.utils.data import random_split
## FFCV imports
from typing import List


# from ffcv.fields import IntField, RGBImageField
# from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
# from ffcv.loader import Loader, OrderOption
# from ffcv.pipeline.operation import Operation
# from ffcv.transforms import RandomHorizontalFlip, Cutout, \
#     RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, RandomResizedCrop, NormalizeImage
# from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
#     RandomResizedCropRGBImageDecoder
# from ffcv.fields.basics import IntDecoder
# from ffcv.transforms.common import Squeeze
# from ffcv.writer import DatasetWriter


# from ffcv.writer import DatasetWriter
# from ffcv.fields import RGBImageField, IntField
# from ffcv.loader import Loader, OrderOption
# from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
# from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder
# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs

# class TinyImageNetaValset(datasets.DatasetFolder):
#     def __init__(self,dir,transformations):
#         super(TinyImageNetaValset, self).__init__()
#
#     def find_classes(self, directory: str) -> typing.Tuple[typing.List[str], typing.Dict[str, int]]:
#

# ImageNet Recipe#################################

# Optimizer & LR scheme
#   ngpus=8,
#   batch_size=32,  # per GPU
#
#   epochs=90,
#   opt='sgd',
#   momentum=0.9,
#
#   lr=0.1,
#   lr_scheduler='steplr',
#   lr_step_size=30,
#   lr_gamma=0.1,
#
#
#   # Regularization
#   weight_decay=1e-4,
#
#
#   # Resizing
#   interpolation='bilinear',
#   val_resize_size=256,
#   val_crop_size=224,
#   train_crop_size=224,


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def model_params(model):
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
    return params


def cal_grad_ACCELERATOR(net: nn.Module, trainloader, device,
                         num_stop=5000, T=1, criterion=nn.CrossEntropyLoss):
    num_data = 0  # count the number of datum points in the dataloader
    base_params = model_params(net)
    gbase = [torch.zeros(p.size()).to(device) for p in base_params]
    for inputs, targets in trainloader:
        if (num_data >= num_stop):
            break
        net.zero_grad()
        tmp_num_data = inputs.size(0)
        outputs = net(inputs) / T
        loss = criterion(outputs, targets)
        gradsH = torch.autograd.grad(loss, base_params, create_graph=False)
        ### update
        gbase = [gbase1 + g1.detach().clone() * float(tmp_num_data) for gbase1, g1 in zip(gbase, gradsH)]
        num_data += float(tmp_num_data)

    gbase = [gbase1 / num_data for gbase1 in gbase]

    return gbase


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct: torch.Tensor = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_layer_dict(model: torch.nn.Module):
    """
    :param model:
    :return: The name of the modules that are not batch norm and their correspondent weight with original shape.
    """
    iter_1 = model.named_modules()
    layer_dict = []

    for name, m in iter_1:
        with torch.no_grad():
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not \
                    isinstance(m, nn.BatchNorm3d):
                layer_dict.append((name, m.weight.data.cpu().detach()))
                # if "shortcut" in name:
                #     print(f"{name}:{m.weight.data}")
                # if not isinstance(m, nn.Conv2d):
                #     print(f"{name}:{m.weight.data}")

    return layer_dict


def get_buffer_dict(model: torch.nn.Module):
    """
    :param model:
    :return: The name of the modules that are not batch norm and their correspondent weight with original shape.
    """
    iter_1 = model.named_modules()
    layer_dict = []

    for name, m in iter_1:
        with torch.no_grad():
            if hasattr(m, 'weight_mask') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not \
                    isinstance(m, nn.BatchNorm3d):
                layer_dict.append((name, m.weight_mak.data.cpu().detach()))
                # if "shortcut" in name:
                #     print(f"{name}:{m.weight.data}")
                # if not isinstance(m, nn.Conv2d):
                #     print(f"{name}:{m.weight.data}")
    if len(layer_dict) == 0:
        raise Exception("Model needs to have weight_maks attributes on modules")
    # assert len(layer_dict)!=0, "Model needs to have weight_maks attributes on modules"
    return layer_dict


def get_mask(model, dense=False):
    if not dense:
        try:
            return dict(get_buffer_dict(model))
        except:
            temp = lambda w: (w != 0).type(torch.float).to('cuda')
            names, weights = zip(*get_layer_dict(model))
            masks = list(map(temp, weights))
            mask_dict = dict(zip(names, masks))
            return mask_dict
    else:
        names, weights = zip(*get_layer_dict(model))
        masks = list(map(torch.ones_like, weights))
        mask_dict = dict(zip(names, masks))
        return mask_dict


def test_pin_and_num_workers(args):
    current_directory = Path().cwd()
    data_path = ""
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        data_path = "/nobackup/sclaam/data/"
    elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
        data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\MNIST"
    elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
        data_path = "datasets/"
    traindir = data_path + 'imagenet/' + 'train'
    testdir = data_path + 'imagenet/' + 'val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    whole_train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    print(f"Length of dataset: {len(whole_train_dataset)}")
    print(args)
    test_dataset = datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    # train_dataset, val_dataset = torch.utils.data.random_split(whole_train_dataset, [1231167, 5000])
    train_dataset, val_dataset = torch.utils.data.random_split(whole_train_dataset,
                                                               [len(whole_train_dataset) - 5000, 5000])
    # big_test,small_test = torch.utils.data.random_split(test_dataset, [len(test_dataset)-5000, 5000])
    my_dataset = val_dataset
    write_path = data_path + "imagenet/valSplit_dataset.beton"

    # For the validation set that I use to recover accuracy

    # # Pass a type for each data field
    # writer = DatasetWriter(write_path, {
    #     # Tune options to optimize dataset size, throughput at train-time
    #     'image': RGBImageField(
    #         max_resolution=256,
    #         jpeg_quality=90
    #     ),
    #     'label': IntField()
    # })
    # # Write dataset
    # writer.from_indexed_dataset(my_dataset)

    # For the validation set that I use to recover accuracy

    import time
    import multiprocessing
    use_cuda = torch.cuda.is_available()
    core_number = multiprocessing.cpu_count()
    batch_size = 64
    train_sampler = None
    best_num_worker = [0, 0]
    best_time = [99999999, 99999999]
    print('cpu_count =', core_number)

    def loading_time(num_workers, pin_memory):
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, **kwargs)
        start = time.time()
        for epoch in range(4):
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx == 15:
                    break
                pass
        end = time.time()
        print("  Used {} second with num_workers = {}".format(end - start, num_workers))
        return end - start

    for pin_memory in [False, True]:
        print("While pin_memory =", pin_memory)
        for num_workers in range(0, core_number * 2 + 1, 4):
            current_time = loading_time(num_workers, pin_memory)
            if current_time < best_time[pin_memory]:
                best_time[pin_memory] = current_time
                best_num_worker[pin_memory] = num_workers
            else:  # assuming its a convex function
                if best_num_worker[pin_memory] == 0:
                    the_range = []
                else:
                    the_range = list(range(best_num_worker[pin_memory] - 3, best_num_worker[pin_memory]))
                for num_workers in (
                        the_range + list(range(best_num_worker[pin_memory] + 1, best_num_worker[pin_memory] + 4))):
                    current_time = loading_time(num_workers, pin_memory)
                    if current_time < best_time[pin_memory]:
                        best_time[pin_memory] = current_time
                        best_num_worker[pin_memory] = num_workers
                break
    if best_time[0] < best_time[1]:
        print("Best num_workers =", best_num_worker[0], "with pin_memory = False")
    else:
        print("Best num_workers =", best_num_worker[1], "with pin_memory = True")
    return


def load_small_imagenet(args: dict, val_size=5000,test_size=10000):

    assert isinstance(args, dict), "args for load_small_imagenet must be a dictionary"

    ratio = 256 / 224

    normalize_train = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                           std=[0.2302, 0.2265, 0.2262])

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4824, 0.4495, 0.3981], [0.2301, 0.2264, 0.2261]),
    ])

    test_dataset = datasets.ImageFolder(args["valdir"], transform=transform_test)

    indices = torch.arange(0, test_size)
    smaller_testset = torch.utils.data.Subset(test_dataset, indices)
    test_loader = torch.utils.data.DataLoader(smaller_testset,
                                              batch_size=args["batch_size"],
                                              num_workers=args["num_workers"],
                                              shuffle=False)

    whole_train_dataset = datasets.ImageFolder(
        args["traindir"],
        transforms.Compose([
            transforms.RandomResizedCrop(args["resolution"]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_train,
        ]))
    current_directory = Path(args["traindir"])

    train_size = 95000
    # val_size = 5000
    if 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
        train_size = 94999
        # val_size = 5000
    elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
        train_size = 94999
        # val_size = 5000

    train_data, val_data = random_split(whole_train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args["batch_size"],
                                               num_workers=args["num_workers"],
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args["batch_size"],
                                             num_workers=args["num_workers"],
                                             shuffle=True)

    return train_loader, val_loader, test_loader


def load_tiny_imagenet(args):
    normalize_train = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                           std=[0.2302, 0.2265, 0.2262])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4824, 0.4495, 0.3981], [0.2301, 0.2264, 0.2261]),
    ])

    test_dataset = datasets.ImageFolder(args["valdir"], transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args["batch_size"],
                                              num_workers=args["num_workers"],
                                              shuffle=False)

    whole_train_dataset = datasets.ImageFolder(
        args["traindir"],
        transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_train,
        ]))

    train_data, val_data = random_split(whole_train_dataset, [95000, 5000])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args["batch_size"],
                                               num_workers=args["num_workers"],
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args["batch_size"],
                                             num_workers=args["num_workers"],
                                             shuffle=True)

    return train_loader, val_loader, test_loader


def change_directory_structure(tiny_imagenet_val_directory):
    annotations_df = pd.read_csv(tiny_imagenet_val_directory + "/val_annotations.txt", sep="\t", header=None,
                                 index_col=False)

    classes = annotations_df[1].unique()
    import os
    import shutil
    for i_class in classes:
        os.mkdir(tiny_imagenet_val_directory + "/{}".format(i_class))
    for name in glob.glob(tiny_imagenet_val_directory + "/images/*.JPEG"):
        file_name = os.path.basename(name)
        sample_class = annotations_df[annotations_df[0] == file_name][1][0]
        shutil.copy(name, tiny_imagenet_val_directory + "/{}".format(sample_class))


def load_imageNet(args):
    current_directory = Path().cwd()
    data_path = ""
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        data_path = "/nobackup/sclaam/data/"
    elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
        data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\MNIST"
    elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
        data_path = "datasets/"
    traindir = data_path + 'imagenet/' + 'train'
    testdir = data_path + 'imagenet/' + 'val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    whole_train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    print(f"Length of dataset: {len(whole_train_dataset)}")
    print(args)
    test_dataset = datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    # train_dataset, val_dataset = torch.utils.data.random_split(whole_train_dataset, [1231167, 5000])
    train_dataset, val_dataset = torch.utils.data.random_split(whole_train_dataset,
                                                               [len(whole_train_dataset) - 5000, 5000])
    # big_test,small_test = torch.utils.data.random_split(test_dataset, [len(test_dataset)-5000, 5000])
    my_dataset = val_dataset
    write_path = data_path + "imagenet/valSplit_dataset.beton"

    # For the validation set that I use to recover accuracy

    # # Pass a type for each data field
    # writer = DatasetWriter(write_path, {
    #     # Tune options to optimize dataset size, throughput at train-time
    #     'image': RGBImageField(
    #         max_resolution=256,
    #         jpeg_quality=90
    #     ),
    #     'label': IntField()
    # })
    # # Write dataset
    # writer.from_indexed_dataset(my_dataset)

    # For the validation set that I use to recover accuracy

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True,
        num_workers=args['num_workers'], pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args['batch_size'], shuffle=True,
        num_workers=args['num_workers'], pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(test_dataset
                                              ,
                                              batch_size=args['batch_size'], shuffle=False,
                                              num_workers=args['num_workers'], pin_memory=True)
    return train_loader, val_loader, test_loader


def main(args):
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-exp', '--experiment',type=int,default=11 ,help='Experiment number', required=True)
    # parser.add_argument('-pop', '--population', type=int,default=1,help = 'Population', required=False)
    # parser.add_argument('-gen', '--generation',type=int,default=10, help = 'Generations', required=False)
    # # parser.add_argument('-mod', '--model_type',type=str,default=alternative, help = 'Type of model to use', required=False)
    # parser.add_argument('-ep', '--epochs',type=int,default=10, help='Epochs for fine tuning', required=False)
    # parser.add_argument('-sig', '--sigma',type=float,default=0.005, help='Noise amplitude', required=True)
    # parser.add_argument('-bs', '--batch_size',type=int,default=512, help='Batch size', required=True)
    # parser.add_argument('-pr', '--pruner',type=str,default="global", help='Type of prune', required=True)
    # parser.add_argument('-dt', '--dataset',type=str,default="cifar10", help='Dataset for experiments', required=True)
    # parser.add_argument('-ar', '--architecture',type=str,default="resnet18", help='Type of architecture', required=True)
    # # parser.add_argument('-so', '--solution',type=str,default="", help='Path to the pretrained solution, it must be consistent with all the other parameters', required=True)
    # parser.add_argument('-mt', '--modeltype',type=str,default="alternative", help='The type of model (which model definition/declaration) to use in the architecture', required=True)
    # parser.add_argument('-pru', '--pruning_rate',type=float,default=0.9, help='percentage of weights to prune', required=False)
    # parser.add_argument('-acc', '--accelerate',type=bool,default=False, help='Use Accelerate package for mixed in precision and multi GPU and MPI library compatibility', required=False)
    #

    # args = vars(parser.parse_args())
    bigining_of_time = time.time()
    # args = {"accelerate": True, 'num_workers': 10}

    solution = ""
    exclude_layers = None
    if args["dataset"] == "cifar100":
        if args["modeltype"] == "alternative":
            if args["architecture"] == "resnet18":
                solution = "trained_modes/cifar100/resnet18_cifar100_traditional_train.pth"
                exclude_layers = ["conv1", "linear"]
            if args["architecture"] == "VGG19":
                solution = "trained_models/cifar100/vgg19_cifar100_traditional_train.pth"
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = "trained_models/cifar100/resnet50_cifar100.pth"
                exclude_layers = ["conv1", "linear"]
    if args["dataset"] == "cifar10":
        if args["modeltype"] == "alternative":
            if args["architecture"] == "resnet18":
                solution = "trained_models/cifar10/resnet18_cifar10_traditional_train_valacc=95,370.pth"
                exclude_layers = ["conv1", "linear"]
            if args["architecture"] == "VGG19":
                solution = "trained_models/cifar100/VGG19_cifar10_traditional_train_valacc=93,57.pth"
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = "trained_models/cifar10/resnet50_cifar10.pth"
                exclude_layers = ["conv1", "linear"]
    if args["dataset"] == "imagenet":

        if args["modeltype"] == "alternative":

            if args["architecture"] == "resnet18":
                solution = ""
                exclude_layers = ["conv1", "linear"]

            if args["architecture"] == "VGG19":
                solution = ""
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = ""
                exclude_layers = ["conv1", "linear"]

        if args["modeltype"] == "hub":

            if args["architecture"] == "resnet18":
                solution = ""
                exclude_layers = ["conv1", "linear"]
                exclude_layers = ["conv1", "fc"]
                # exclude_layers = []
            if args["architecture"] == "VGG19":
                raise NotImplementedError("Not implemented")
                solution = ""
                exclude_layers = ["features.0", "classifier"]
            if args["architecture"] == "resnet50":
                solution = ""
                exclude_layers = ["conv1", "fc"]

    cfg = omegaconf.DictConfig({
        "population": args["population"],
        "generations": 10,
        "epochs": args["epochs"],
        "short_epochs": 10,
        "architecture": args["architecture"],
        "solution": "",
        "noise": "gaussian",
        "pruner": args["pruner"],
        "model_type": args["modeltype"],
        "fine_tune_exclude_layers": True,
        "fine_tune_non_zero_weights": True,
        "sampler": "tpe",
        "flop_limit": 0,
        "one_batch": True,
        "measure_gradient_flow": True,
        "full_fine_tune": False,
        "use_stochastic": True,
        "sigma": args["sigma"],
        "noise_after_pruning": 0,
        "amount": args["pruning_rate"],
        "dataset": args["dataset"],
        "batch_size": args["batch_size"],
        "num_workers": args["num_workers"],
        "save_model_path": "stochastic_pruning_models/",
        "save_data_path": "stochastic_pruning_data/",
        "gradient_cliping": True,
        "use_wandb": True
    })
    cfg.exclude_layers = exclude_layers
    # cfg = omegaconf.DictConfig({
    #     "dataset": "cifar10",
    #     "batch_size": 64,
    #     "num_workers": 0,
    #     "fine_tune_exclude_layers": True,
    #     "fine_tune_non_zero_weights": True,
    #     "exclude_layers": []
    #
    #
    # })

    if cfg.use_wandb:
        os.environ["wandb_start_method"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            project="stochastic_pruning",
            name=f"Training_ImageNet",
            notes="This is for imageNet training",
            reinit=True,
        )
    train_loader, val_loader, test_loader = load_imageNet(args)
    print("Create the datasets")
    # train_loader, val_loader, test_loader = get_cifar_datasets(cfg)
    net = get_model(cfg)
    # disable_bn(net)
    # disable_bn(net2)

    # in_features = net2.fc.in_features
    #
    # net2.fc = nn.Linear(in_features, 10)
    # # j
    # in_features = net.fc.in_features
    #
    # net.fc = nn.Linear(in_features, 10)

    # net.load_state_dict(torch.load("/nobackup/sclaam/trained_models/resnet50_imagenet.pth"))
    # prune_with_rate(net, 0.5, type="global")
    # remove_reparametrization(net)
    #
    # mask = get_mask(model=net)
    # apply_mask_with_hook(net, mask)

    # print("Sparsity of model before \"prepare\": {}".format(sparsity(net)))

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    accelerator = None
    if args["accelerate"]:
        accelerator = Accelerator(mixed_precision="fp16")
        net, optimizer, val_loader_accel, lr_scheduler, test_loader_accel = accelerator.prepare(
            net, optimizer, val_loader, lr_scheduler, test_loader
        )
    eval_set = None
    print("Passed accelerating thing")
    state = AcceleratorState()
    print("mixed precision?: {}".format(state.mixed_precision))
    # net.cuda()
    # t0 = time.time()
    # data, y = next(iter(val_loader_accel))
    # _, unit_sparse_flops = flops(net, data)
    # t1 = time.time()
    # print("Time for calculating batch flops: {}s".format(t1 - t0))
    # net.train()

    # t0 = time.time()
    #
    # measure_and_record_gradient_flow_with_ACCELERATOR(net, accelerator, val_loader_accel, test_loader_accel,
    #                                                   "imagenet_measure_record_test.csv", total_sparse_flops, 0,
    #                                                   use_wandb=False,criterion=criterion)
    #
    # t1 = time.time()
    # total_time = t1 - t0
    # hours_float = total_time / 3600
    # decimal_part = hours_float - total_time // 3600
    # minutes = decimal_part * 60
    # print("Time for measure gradient with accelerator: {}h and {} minutes".format(total_time // 3600, minutes))

    t0 = time.time()
    # net2.load_state_dict(accelerator.unwrap_model(net).state_dict())
    # measure_and_record_gradient_flow(accelerator.unwrap_model(net), val_loader, test_loader,cfg = cfg,
    #                                                   filepath="imagenet_measure_record_test.csv", total_flops=total_sparse_flops, epoch=0,
    #                                                   use_wandb=False,mask_dict=mask)
    t1 = time.time()
    total_time = t1 - t0
    hours_float = total_time / 3600
    decimal_part = hours_float - total_time // 3600
    minutes = decimal_part * 60
    print("Time for measure gradient WITHOUT accelerator: {}h and {} minutes".format(total_time // 3600, minutes))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    begining = time.time()
    best_accuracy = -1
    end = time.time()
    for e in range(args["epochs"]):
        for batch_idx, (data, target) in enumerate(val_loader_accel):
            if not args["accelerate"]:
                data, target = data.cuda(), target.cuda()
            # switch to train mode
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            output = net(data)

            # print(
            #     "Sparsity of model after being unwrapped and forward call done (possibly acctivating the pre-hook): {}".format(
            #         sparsity(unwraped_model)))
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            if args["accelerate"]:
                accelerator.backward(loss)
            else:
                loss.backward()
            # mask_gradient(net, mask_dict=mask)

            accelerator.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            # apply_mask(net,mask_dict=mask)
            # total_sparse_flops += 3 * unit_sparse_flops

            # accelerator.prepare()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 10 == 0:
                unwraped_model = accelerator.unwrap_model(net)
                print("Sparsity of model after being unwrapped and gradients applied: {}".format(
                    sparsity(unwraped_model)))
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    e, batch_idx, len(val_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        lr_scheduler.step()

        test_accuracy = test_with_accelerator(net, test_loader_accel, accelerator=accelerator)

        print("Test Accuracy: {}".format(test_accuracy))

        if cfg.use_wandb:
            wandb.log({"test_accuracy": test_accuracy})

        if best_accuracy < test_accuracy:
            unwraped_model = accelerator.unwrap_model(net)

            state_dict = unwraped_model.state_dict()

            save = {"net": state_dict, "test_accuracy": test_accuracy, "epoch": e}

            temp_name = args["file_path"] + "{}_{}_{}_dense.pth".format(args["architecture"], args["dataset"],
                                                                        args["modeltype"])

            torch.save(save, temp_name)

            best_accuracy = test_accuracy

            del unwraped_model

    total_time = time.time() - begining
    hours_float = total_time / 3600
    decimal_part = hours_float - total_time // 3600
    minutes = decimal_part * 60
    print("Total time was {} hours and {} minutes".format(total_time // 3600, minutes))

    #####################################################################
    end_of_time = time.time()
    total_time = end_of_time - bigining_of_time
    hours_float = total_time / 3600
    decimal_part = hours_float - total_time // 3600
    minutes = decimal_part * 60

    print("Total execution time: {}h and {} minutes".format(total_time // 3600, minutes))


def test_num_workers():
    from time import time
    import multiprocessing as mp

    print("mp.cpu_count(): {}".format(mp.cpu_count()))
    list_num_workers = [0, 18, 20, 22, 24]
    for num_workers in list_num_workers:

        args = {'num_workers': num_workers}
        train_loader, val_loader, test_loader = load_imageNet(args)
        accelerator = Accelerator()
        val_loader = accelerator.prepare(
            val_loader
        )
        limit = 100
        start = time()
        for i, data in enumerate(val_loader, 0):
            if i < limit:

                pass

            else:

                break

        end = time()

        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stochastic pruning experiments')
    parser.add_argument('-pop', '--population', type=int, default=1, help='Population', required=False)
    parser.add_argument('-gen', '--generation', type=int, default=10, help='Generations', required=False)
    # parser.add_argument('-mod', '--model_type',type=str,default=alternative, help = 'Type of model to use', required=False)
    # parser.add_argument('-ep', '--epochs', type=int, default=10, help='Epochs for fine tuning', required=True)
    parser.add_argument('-sig', '--sigma', type=float, default=0.005, help='Noise amplitude', required=False)
    # parser.add_argument('-bs', '--batch_size', type=int, default=512, help='Batch size', required=True)
    parser.add_argument('-pr', '--pruner', type=str, default="global", help='Type of prune', required=False)
    # parser.add_argument('-dt', '--dataset', type=str, default="cifar10", help='Dataset for experiments', required=True)
    # parser.add_argument('-ar', '--architecture', type=str, default="resnet18", help='Type of architecture',
    #                     required=True)
    # parser.add_argument('-so', '--solution',type=str,default="", help='Path to the pretrained solution, it must be consistent with all the other parameters', required=True)
    # parser.add_argument('-mt', '--modeltype', type=str, default="alternative",
    #                     help='The type of model (which model definition/declaration) to use in the architecture',
    #                     required=True)
    parser.add_argument('-pru', '--pruning_rate', type=float, default=0.9, help='percentage of weights to prune',
                        required=False)
    parser.add_argument('-nw', '--num_workers', type=int, default=4, help='Number of workers', required=False)
    parser.add_argument('-ngpu', '--number_gpus', type=int, default=4, help='Number of workers', required=False)
    parser.add_argument('-acc', '--accelerate', type=bool, default=False,
                        help='Use Accelerate package for mixed in precision and multi GPU and MPI library compatibility',
                        required=False)
    parser.add_argument('-fp', '--file_path', type=str, default="/nobackup/sclaam/trained_models/",
                        help='Location where the models is going to be saved', required=False)
    parser.add_argument('--traindir', type=str, default="/home/luisaam/Documents/PhD/data/tiny_imagenet_200/train",
                        help='Number of workers', required=False)
    parser.add_argument('--valdir', type=str, default="/home/luisaam/Documents/PhD/data/tiny_imagenet_200/val",
                        help='Number of workers', required=False)

    args = vars(parser.parse_args())
    # LeMain(args)
    # main(args)
    load_tiny_imagenet(args)
    # test_num_workers()
