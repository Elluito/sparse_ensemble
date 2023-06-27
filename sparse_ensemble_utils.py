import copy
import logging
import time
import typing
from functools import partial
from typing import TYPE_CHECKING
import omegaconf
# from main import get_layer_dict
import optuna.samplers
import pandas as pd
from pathlib import Path
from einops import repeat
import numpy as np
import torch
from torch import nn
from shrinkbench.metrics.flops import flops
from torchmetrics import Accuracy
import wandb
from decimal import Decimal
from flowandprune.imp_estimator import cal_grad,cal_grad_fisher,cal_hg
import pyhessian as pyhes
from torch.nn.utils import vector_to_parameters,parameters_to_vector
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: {}".format(device))
def test(net, use_cuda, testloader, one_batch=False, verbose=2, count_flops=False, batch_flops=0):
    if use_cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    sparse_flops = 0
    first_time = 1
    sparse_flops_batch = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if count_flops:
                sparse_flops += batch_flops
            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % 100 == 0:
                if verbose == 2:
                    print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                          % (test_loss / (batch_idx + 1), 100. * correct.item() / total, correct, total))
            if one_batch:
                if count_flops:
                    return 100. * correct.item() / total, sparse_flops
                else:
                    return 100. * correct.item() / total
    if verbose == 1 or verbose == 2:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss / len(testloader), correct, total,
            100. * correct.item() / total))
    net.cpu()
    if count_flops:
        return 100. * correct.item() / total, sparse_flops
    else:
        return 100. * correct.item() / total


def get_random_batch(dataLoader):
    N = len(dataLoader)
    iterator = iter(dataLoader)
    random_int = np.random.randint(N) + 1
    batch = None
    for i in range(random_int):
        batch = next(iterator)
    return batch


def disable_bn(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) \
                or isinstance(module, nn.BatchNorm3d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def disable_exclude_layers(model: nn.Module, exclude_layers=[]):
    dict_of_modules = dict(list(model.named_modules()))
    for name in exclude_layers:
        dict_of_modules[name].eval()
        for param in dict_of_modules[name].parameters():
            param.requires_grad = False


def disable_all_except(model: nn.Module, exclude_layers=[]):
    dict_of_modules = dict(list(model.named_modules()))
    for name, module in dict_of_modules.items():
        if name in exclude_layers:
            for param in module.parameters():
                param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = False


# def get_mask(weight: torch.FloatTensor):
#     return (weight != 0).type(torch.float)

@torch.no_grad()
def mask_gradient(model: torch.nn.Module, mask_dict: dict):
    parameters_dict = dict(model.named_parameters())
    # for name,parameter in parameters_dict.items():
    #     for mask_name in mask_dict.keys():
    #         if mask_name in name:
    #             parameter.grad.data.mul_(mask_dict[mask_name].to("cuda"))
    for name, module in model.named_modules():
        if name in mask_dict.keys():
            if hasattr(module.weight, "grad"):
                if module.weight.grad is not None:
                    # print("Module Name: {}".format(name))
                    module.weight.grad.mul_(mask_dict[name].to("cuda"))


def efficient_population_evaluation(memory: list, model: nn.Module, image, use_cuda: bool, dataLoader=None):
    x, y = get_random_batch(dataLoader)
    if use_cuda:
        x, y = x.cuda(), y.cuda()
    rand_index = np.random.randint(0, len(x), size=1)
    prediction = model(x[rand_index]).detach()
    if prediction.eq(y[rand_index].data):
        memory.append(model)
        return True
    else:
        return False


def get_random_image_label(dataloader):
    x, y = get_random_batch(dataloader)
    rand_index = np.random.randint(0, len(x), size=1)
    image = x[rand_index]
    return image, y[rand_index]


def check_for_layers_collapse(model):
    names, weights = zip(*get_layer_dict(model))
    for indx, w in enumerate(weights):
        if torch.count_nonzero(w) == 0:
            raise Exception("Layer {} has 0 weights different form 0 the layer has collapsed".format(names[indx]))

#COMMENT: THIS IS FOR THE EXPERIMENTS OF FINE-TUNING UNRESTRICTED AN then compare the final solution to the deterministic to see if they are on a different basin still.
def unrestricted_fine_tune_measure_flops(pruned_model: nn.Module, dataLoader: torch.utils.data.DataLoader,
                                       testLoader: torch.utils.data.DataLoader,
                                       epochs=1,
                                       FLOP_limit: float = 0, initial_flops=0, use_wandb=False, exclude_layers=[],
                                       fine_tune_exclude_layers=False, fine_tune_non_zero_weights=True,
                                       gradient_flow_file_prefix="", cfg=None,):
    # optimizer = torch.optim.SGD()
    optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.0001,
                                momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    grad_clip = 0
    if cfg.gradient_cliping:
        grad_clip = 0.1
    names, weights = zip(*get_layer_dict(pruned_model))
    if cfg.dataset == "cifar10" or cfg.dataset == "mnist":
        accuracy = Accuracy(task="multiclass", num_classes=10).to("cuda")
    if cfg.dataset == "cifar100":
        accuracy = Accuracy(task="multiclass", num_classes=100).to("cuda")
    if cfg.dataset == "imagenet":
        accuracy = Accuracy(task="multiclass", num_classes=1000).to("cuda")

    mask_dict = get_mask(model=pruned_model)
    for name in exclude_layers:
        if name in list(mask_dict.keys()):
            mask_dict.pop(name)
    total_FLOPS = initial_flops
    total_sparse_FLOPS = initial_flops
    # This is for the first batch of the training. The forward pass is sparse but the backward pass is dense and then subsequent
    #  forward and backward passes are dense.
    first_time = 1

    #TODO: Here I need to be carefull of how I do the recording since this model is unrestricted
    data, y = next(iter(dataLoader))
    forward_pass_dense_flops, forward_pass_sparse_flops = flops(pruned_model, data)

    file_path = None
    weights_path = ""
    if gradient_flow_file_prefix != "":
        file_path = gradient_flow_file_prefix
        file_path +=  "recordings.csv"

        if  Path(gradient_flow_file_prefix).owner() == "sclaam":
            weights_file_path = "/nobackup/sclaam/" + gradient_flow_file_prefix + "weigths/"
        if Path(gradient_flow_file_prefix).owner() == "luisaam":
            weights_file_path =  "GF_data/"+ gradient_flow_file_prefix + "weigths/"

        weights_path = Path(weights_file_path)
        weights_path .mkdir(parents=True)
        measure_and_record_gradient_flow(pruned_model,dataLoader,testLoader,cfg,file_path,total_FLOPS+forward_pass_sparse_flops+2*forward_pass_sparse_flops,-1,mask_dict=mask_dict,use_wandb=use_wandb)

    pruned_model.cuda()
    pruned_model.train()
    disable_bn(pruned_model)
    if not fine_tune_exclude_layers:
        disable_exclude_layers(pruned_model, exclude_layers)
    if not fine_tune_non_zero_weights:
        disable_all_except(pruned_model, exclude_layers)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataLoader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # first forward-backward step
            predictions = pruned_model(data)
            # enable_bn(model)
            loss = criterion(predictions, target)
            loss.backward()
            backward_flops_sparse = 2 * forward_pass_sparse_flops
            backward_flops_dense = 2 * forward_pass_dense_flops
            batch_dense_flops = forward_pass_dense_flops + backward_flops_dense
            batch_sparse_flops = forward_pass_sparse_flops + backward_flops_sparse
            total_FLOPS += batch_dense_flops
            total_sparse_FLOPS += batch_sparse_flops
            accuracy.update(preds=predictions.cuda(), target=target.cuda())
            # mask_gradient(pruned_model, mask_dict=mask_dict)

            if grad_clip:
                nn.utils.clip_grad_value_(pruned_model.parameters(), grad_clip)

            optimizer.step()
            lr_scheduler.step()
            if use_wandb:
                acc = accuracy.compute()
                test_accuracy = test(pruned_model, use_cuda=True, testloader=[get_random_batch(testLoader)],
                                     one_batch=True)
                wandb.log({
                    "val_set_accuracy": acc * 100,
                    "dense_flops": total_FLOPS,
                    "test_set_accuracy": test_accuracy,
                    "sparsity": sparsity(pruned_model)
                })
            if batch_idx % 10 == 0 or FLOP_limit != 0:
                acc = accuracy.compute()
                flops_sparse = '%.3E' % Decimal(total_FLOPS)
                print(f"Fine-tune Results - Epoch: {epoch}  Avg accuracy: {acc:.2f} Avg loss:"
                      f" {loss.item():.2f} FLOPS:{flops_sparse} sparsity {sparsity(pruned_model) :.3f}")

                if FLOP_limit != 0 and FLOP_limit > total_FLOPS:
                    break
        # if gradient_flow_file_prefix != "":

        if epoch%10 == 0 and gradient_flow_file_prefix != "":
            measure_and_record_gradient_flow(pruned_model,dataLoader,testLoader,cfg,file_path,total_FLOPS,epoch,
                                             mask_dict=mask_dict
                                             ,use_wandb=use_wandb)
            state_dict = pruned_model.state_dict()
            temp_name = weights_path / "epoch_{}.pth".format(epoch)
            torch.save(state_dict,temp_name)
        if FLOP_limit != 0:
            if total_FLOPS > FLOP_limit:
                break
    if gradient_flow_file_prefix != "":

        measure_and_record_gradient_flow(pruned_model,dataLoader,testLoader,cfg,file_path,total_FLOPS,epochs,
                                         mask_dict=mask_dict
                                         ,use_wandb=use_wandb)

    test_set_performance = test(pruned_model, use_cuda=True, testloader=testLoader)

    if use_wandb:
        if gradient_flow_file_prefix != "":
            df = pd.read_csv(file_path,sep=",", header=0 , index_col=False)
            table = wandb.Table(data=df)
            wandb.log({"Gradient Flow results": table})
        wandb.log({
            "test_set_accuracy": test_set_performance,
            "dense_flops": total_FLOPS,
            "final_accuracy": test_set_performance
        })


    return total_FLOPS
    # msg_perormance = f"{performance:.2f}".replace(".", ",")


def restricted_fine_tune_measure_flops(pruned_model: nn.Module, dataLoader: torch.utils.data.DataLoader,
                                       testLoader: torch.utils.data.DataLoader,
                                       epochs=1,
                                       FLOP_limit: float = 0, initial_flops=0, use_wandb=False, exclude_layers=[],
                                       fine_tune_exclude_layers=False, fine_tune_non_zero_weights=True,
                                       gradient_flow_file_prefix="", cfg=None):
    # optimizer = torch.optim.SGD()
    #################### Best accuracy yet ################################


    ####################
    optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.0001,
                                momentum=0.9, weight_decay=5e-4)
    if "cifar" in cfg.dataset:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    grad_clip = 0
    if cfg.gradient_cliping:
        grad_clip = 0.1
    names, weights = zip(*get_layer_dict(pruned_model))
    if cfg.dataset == "cifar10" or cfg.dataset == "mnist":
        accuracy = Accuracy(task="multiclass", num_classes=10).to("cuda")
    if cfg.dataset == "cifar100":
        accuracy = Accuracy(task="multiclass", num_classes=100).to("cuda")
    if cfg.dataset == "imagenet":
        accuracy = Accuracy(task="multiclass", num_classes=1000).to("cuda")

    mask_dict = get_mask(model=pruned_model)
    for name in exclude_layers:
        if name in list(mask_dict.keys()):
            mask_dict.pop(name)
    total_FLOPS = 0
    total_sparse_FLOPS = initial_flops
    # first_time = 1


    data, y = next(iter(dataLoader))
    forward_pass_dense_flops, forward_pass_sparse_flops = flops(pruned_model, data)

    file_path = None
    weights_path = ""
    if gradient_flow_file_prefix != "":
        file_path = gradient_flow_file_prefix
        file_path +=  "recordings.csv"

        if Path(gradient_flow_file_prefix).owner() == "sclaam":
            weights_file_path = "/nobackup/sclaam/" + gradient_flow_file_prefix + "weigths/"
        if Path(gradient_flow_file_prefix).owner() == "luisaam":
            weights_file_path =  "GF_data/"+ gradient_flow_file_prefix + "weigths/"

        weights_path = Path(weights_file_path)
        weights_path .mkdir(parents=True)
        measure_and_record_gradient_flow(pruned_model,dataLoader,testLoader,cfg,file_path,total_sparse_FLOPS,-1,mask_dict=mask_dict,use_wandb=use_wandb)
        # state_dict = pruned_model.state_dict()
        # temp_name = weights_path / "epoch_{}.pth".format(-1)
        # torch.save(state_dict,temp_name)

    pruned_model.cuda()
    pruned_model.train()
    disable_bn(pruned_model)
    if not fine_tune_exclude_layers:
        disable_exclude_layers(pruned_model, exclude_layers)
    if not fine_tune_non_zero_weights:
        disable_all_except(pruned_model, exclude_layers)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataLoader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # first forward-backward step
            predictions = pruned_model(data)
            # enable_bn(model)
            loss = criterion(predictions, target)
            loss.backward()
            backward_flops_sparse = 2 * forward_pass_sparse_flops
            backward_flops_dense = 2 * forward_pass_dense_flops
            batch_dense_flops = forward_pass_dense_flops + backward_flops_dense
            batch_sparse_flops = forward_pass_sparse_flops + backward_flops_sparse
            total_FLOPS += batch_dense_flops
            total_sparse_FLOPS += batch_sparse_flops
            accuracy.update(preds=predictions.cuda(), target=target.cuda())
            # Mask the grad_
            mask_gradient(pruned_model, mask_dict=mask_dict)

            if grad_clip:
                nn.utils.clip_grad_value_(pruned_model.parameters(), grad_clip)

            optimizer.step()
            lr_scheduler.step()

            # W&B Logging
            if use_wandb:
                acc = accuracy.compute()
                test_accuracy = test(pruned_model, use_cuda=True, testloader=[get_random_batch(testLoader)],
                                     one_batch=True)
                wandb.log({
                    "val_set_accuracy": acc * 100,
                    "sparse_flops": total_sparse_FLOPS,
                    "test_set_accuracy": test_accuracy,
                    "sparsity": sparsity(pruned_model)
                })
            if batch_idx % 10 == 0 or FLOP_limit != 0:
                acc = accuracy.compute()
                flops_sparse = '%.3E' % Decimal(total_sparse_FLOPS)
                print(f"Fine-tune Results - Epoch: {epoch}  Avg accuracy: {acc:.2f} Avg loss:"
                      f" {loss.item():.2f} FLOPS:{flops_sparse} sparsity {sparsity(pruned_model) :.3f}")

                if FLOP_limit != 0 and FLOP_limit > total_sparse_FLOPS:
                    break
        # if gradient_flow_file_prefix != "":

        if epoch%10 == 0 and gradient_flow_file_prefix != "":
            measure_and_record_gradient_flow(pruned_model,dataLoader,testLoader,cfg,file_path,total_sparse_FLOPS,epoch,
                                             mask_dict=mask_dict
                                             ,use_wandb=use_wandb)

            # state_dict = pruned_model.state_dict()
            # temp_name = weights_path / "epoch_{}.pth".format(epoch)
            # torch.save(state_dict,temp_name)
        if FLOP_limit != 0:
            if total_sparse_FLOPS > FLOP_limit:
                break
    if gradient_flow_file_prefix != "":

        measure_and_record_gradient_flow(pruned_model,dataLoader,testLoader,cfg,file_path,total_sparse_FLOPS,epochs,
                                         mask_dict=mask_dict
                                         ,use_wandb=use_wandb)
        state_dict = pruned_model.state_dict()
        temp_name = weights_path / "epoch_{}.pth".format(epochs-1)
        torch.save(state_dict,temp_name)

    test_set_performance = test(pruned_model, use_cuda=True, testloader=testLoader)

    if use_wandb:
        if gradient_flow_file_prefix != "":
            df = pd.read_csv(file_path,sep=",", header=0 , index_col=False)
            table = wandb.Table(data=df)
            wandb.log({"Gradient Flow results": table})
        wandb.log({
            "test_set_accuracy": test_set_performance,
            "sparse_flops": total_sparse_FLOPS,
            "final_accuracy": test_set_performance
        })


    return total_sparse_FLOPS
#
# from .counting.ops import get_inference_FLOPs
#
# # Sparse learning funcs
# from .funcs.grow import registry as grow_registry
# from .funcs.init_scheme import registry as init_registry
# from .funcs.prune import registry as prune_registry
# from .funcs.redistribute import registry as redistribute_registry
# from .sparse_ensemble_utils.smoothen_value import AverageValue
def is_prunable_module(m: torch.nn.Module):
    return (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d))


def get_sampler(cfg: omegaconf.DictConfig):
    if cfg.sampler == "tpe":
        return optuna.samplers.TPESampler()
    if cfg.sampler == "cmaes":
        return optuna.samplers.CmaEsSampler(
            restart_strategy="ipop",
            inc_popsize=2,
            n_startup_trials=10

        )
    if cfg.sampler == "qmc":
        return optuna.samplers.QMCSampler()
    raise NotImplementedError("Sampler {} is not supported yet".format(cfg.sampler))


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


def get_mask(model,dense=False):
    if not dense:
        try:
            return dict(get_buffer_dict(model))
        except:
            temp = lambda w: (w != 0).type(torch.float)
            names, weights = zip(*get_layer_dict(model))
            masks = list(map(temp, weights))
            mask_dict = dict(zip(names, masks))
            return mask_dict
    else:
        names, weights = zip(*get_layer_dict(model))
        masks = list(map(torch.ones_like, weights))
        mask_dict = dict(zip(names, masks))
        return mask_dict

@torch.no_grad()
def apply_mask(model:nn.Module,mask_dict:dict):
    for name,module in model.named_modules():
        if name in mask_dict.keys():
            module.weight.data.mul_(mask_dict[name])
@torch.no_grad()
def apply_mask_with_hook(model:nn.Module,mask_dict:dict):
    '''

    @param model: model to mask
    @param mask_dict:dict of module names and mask
    @return:
    '''
    for name, module in model.named_modules():
        if name in mask_dict.keys():
            def hook(module, grad_input, grad_output) -> tuple[torch.Tensor] or None:
                module.weight.data.mul_(mask_dict[name])
            module.register_forward_pre_hook(hook)
# Function taken from https://github.com/varun19299/rigl-reproducibility/blob/master/sparselearning/utils/ops.py
def random_perm(a: torch.Tensor) -> torch.Tensor:
    """
    Random shuffle a tensor.

    :param a: input Tensor
    :type a: torch.Tensor
    :return: shuffled Tensor
    :rtype: torch.Tensor
    """
    idx = torch.randperm(a.nelement())
    return a.reshape(-1)[idx].reshape(a.shape)


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def get_percentile_per_layer(model: torch.nn.Module, percentile: float = 0.1):
    return_dict = {}
    for name, module in model.named_modules():
        if is_prunable_module(module):
            flat_weights = torch.abs(module.weight.detach().flatten())
            desired_quantile = float(torch.quantile(flat_weights, percentile))
            return_dict[name] = desired_quantile
    return return_dict


def sparsity(model):
    #
    total_params = count_parameters(model)
    non_zero_param = 0
    for name, module in model.named_modules():
        if is_prunable_module(module):

            if list(module.buffers()):
                list_buffers = list(module.buffers())
                non_zero_param += len(list_buffers[0].flatten().nonzero())
            else:

                non_zero_param += len(module.weight.data.flatten().nonzero())
    return 1 - non_zero_param / total_params


# Functions adapted from https://github.com/varun19299/rigl-reproducibility/blob/master/sparselearning/funcs
# /init_scheme.py


def erdos_renyi_per_layer_pruning_rate(model: torch.nn.Module, cfg: omegaconf.DictConfig, is_kernel:
bool = True,
                                       **kwargs) -> typing.Tuple[dict, dict, float, int]:
    names, weights = zip(*get_layer_dict(model))
    prob_dict = get_erdos_renyi_dist(model, names, weights, cfg, is_kernel)
    mask_dict = {}
    pruning_rate_per_layer = {}
    baseline_nonzero = 0
    total_params = 0
    for name, weight in zip(names, weights):
        if name in cfg.exclude_layers:
            continue
        prob = prob_dict[name]
        logging.debug(f"ERK {name}: {weight.shape} prob {prob:.4f}")

        mask_dict[name] = (torch.rand(weight.shape) < prob).float().data
        pruning_rate_per_layer[name] = ((mask_dict[name] == 0).sum() / mask_dict[name].nelement()).item()
        baseline_nonzero += (mask_dict[name] != 0).sum().int().item()
        total_params += weight.numel()

    return mask_dict, pruning_rate_per_layer, baseline_nonzero, total_params


@torch.no_grad()
def get_gradient_norm(model: nn.Module, masked=False):
    sum_of_gradients = 0
    for m in model.modules():
        if is_prunable_module(m):
            if hasattr(m.weight, "grad") and m.weight.grad is not None:
                sum_of_gradients += torch.pow(m.weight.grad, 2).sum().detach().cpu().numpy()

    return np.sqrt(sum_of_gradients)


def measure_and_record_gradient_flow(model: nn.Module, dataLoader, testLoader, cfg, filepath, total_flops, epoch,
                                     mask_dict, use_wandb=False):
    model = copy.deepcopy(model)
    disable_bn(model)
    if cfg.fine_tune_exclude_layers:
        disable_exclude_layers(model, cfg.exclude_layers)
    if cfg.fine_tune_non_zero_weights:
        disable_all_except(model, cfg.exclude_layers)
    model.to(device=device)

    # Calculate everything with respect to the validation set
    val_dict = {}
    grad:typing.List[torch.Tensor] = cal_grad(model,trainloader=dataLoader)
    #
    # if cfg.dataset == "cifar10" or cfg.dataset == "mnist":
    #     hg :typing.List[torch.Tensor] = cal_hg(model,trainloader=dataLoader,n_classes=10)
    # if cfg.dataset == "cifar100":
    #     hg :typing.List[torch.Tensor] = cal_hg(model,trainloader=dataLoader,n_classes=100)
    # if cfg.dataset == "imagenet":
    #     hg :typing.List[torch.Tensor] = cal_hg(model,trainloader=dataLoader,n_classes=1000)

    grad_vect = parameters_to_vector(grad)
    # hg_vect = parameters_to_vector(hg)
    norm_grad = torch.norm(grad_vect)
    # norm_hg = torch.norm(hg_vect)
    val_dict["val_set_gradient_magnitude"] = [float(norm_grad.cpu().detach().numpy())]
    # val_dict["val_set_Hg_magnitude"] = [float(norm_hg.cpu().detach().numpy())]
    # criterion = nn.CrossEntropyLoss()
    # hessian_comp = pyhes.hessian(model,
    #                            criterion,
    #                            dataloader=dataLoader,
    #                            cuda=True if device == "cuda" else False)
    accuracy = test(model, True if device == "cuda" else False,dataLoader, verbose=0)
    model.to(device)
    # print("Calculating eigenvalues on validation set for epoch:{}".format(epoch))
    # top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=5,maxIter=20)
    # print("Calculating hessian trace on validation set for epoch:{}".format(epoch))
    # trace = hessian_comp.trace(maxIter=20)
    # for i,value in enumerate(top_eigenvalues):
    #     val_dict["val_set_EV{}".format(i)] = [value]
    # val_dict["val_set_trace"] = [trace]
    # # density_eigen, density_weight = hessian_comp.density()
    val_dict["val_accuracy"] =  [accuracy]


    # Calculate everything with respect to the test set
    test_dict = {}
    grad:typing.List[torch.Tensor] = cal_grad(model,trainloader=testLoader)
    # t0 = time.time()
    # if cfg.dataset == "cifar10" or cfg.dataset == "mnist":
    #     hg :typing.List[torch.Tensor] = cal_hg(model,trainloader=dataLoader,n_classes=10)
    # if cfg.dataset == "cifar100":
    #     hg :typing.List[torch.Tensor] = cal_hg(model,trainloader=dataLoader,n_classes=100)
    #
    # if cfg.dataset == "imagenet":
    #     hg :typing.List[torch.Tensor] = cal_hg(model,trainloader=dataLoader,n_classes=1000)
    # t1 = time.time()
    # print("Time to calculate Hg: {} s".format(t1-t0))
    grad_vect = parameters_to_vector(grad)
    # hg_vect = parameters_to_vector(hg)
    norm_grad = torch.norm(grad_vect)
    # norm_hg = torch.norm(hg_vect)
    test_dict["test_set_gradient_magnitude"] = [float(norm_grad.cpu().detach().numpy())]
    # test_dict["test_set_Hg_magnitude"] = [float(norm_hg.cpu().detach().numpy())]

    # criterion = nn.CrossEntropyLoss()
    # hessian_comp = pyhes.hessian(model,
    #                              criterion,
    #                              dataloader=testLoader,
    #                              cuda=True if device == "cuda" else False)
    #
    #
    accuracy = test(model, True, testLoader, verbose=0)
    model.to(device)
    # print("Calculating eigenvalues on test set for epoch:{}".format(epoch))
    # start = time.time()
    # top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=2,maxIter=10)
    # stop = time.time()
    # print("Time elapsed: {}s".format(stop-start))
    # print("Calculating hessian trace on test set for epoch:{}".format(epoch))
    # start = time.time()
    # trace = hessian_comp.trace(maxIter=10)
    # stop = time.time()
    # print("Time elapsed: {}s".format(stop-start))
    # for i,value in enumerate(top_eigenvalues):
    #     test_dict["test_set_EV{}".format(i)] = [value]
    # test_dict["test_set_trace"] = [trace]
    test_dict["test_accuracy"] = [accuracy]
    print("Test dictionary :\n {}".format(test_dict))

    # print("accuracy:{}, gradient norm: {},Hg norm {}".format(accuracy,norm_grad,norm_hg))

    if Path(filepath).is_file():
        log_dict = {"Epoch": [epoch], "sparse_flops": [total_flops]}
        log_dict.update(val_dict)
        log_dict.update(test_dict)
        df = pd.DataFrame(log_dict)
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        # Try to read the file to see if it is
        log_dict = {"Epoch": [epoch], "sparse_flops": [total_flops]}
        log_dict.update(val_dict)
        log_dict.update(test_dict)
        df = pd.DataFrame(log_dict)
        df.to_csv(filepath, sep=",", index=False)
    if use_wandb:
        log_dict = {"Epoch": epoch, "sparse_flops": total_flops}
        for n,v in val_dict.items():
            log_dict[n]=v[0]
        for n,v in test_dict.items():
            log_dict[n]=v[0]
        log_dict.update(val_dict)
        log_dict.update(test_dict)
        wandb.log(log_dict)

def get_erdos_renyi_dist(
        model, names, weights, cfg: omegaconf.DictConfig, is_kernel: bool = True
) -> "typing.Dict[str, float]":
    """
    Get layer-wise densities distributed according to
    ER or ERK (erdos-renyi or erdos-renyi-kernel).

    Ensures resulting densities do not cross 1
    for any layer.

    :param masking: Masking instance
    :param is_kernel: use ERK (True), ER (False)
    :return: Layer-wise density dict
    """
    # Same as Erdos Renyi with modification for conv
    # initialization used in sparse evolutionary training
    # scales the number of non-zero weights linearly proportional
    # to the product of all dimensions, that is input*output
    # for fully connected layers, and h*w*in_c*out_c for conv
    # layers.
    _erk_power_scale = 1.0

    epsilon = 1.0
    is_epsilon_valid = False
    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    _dense_layers = set()
    while not is_epsilon_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.
        # We want the total number of connections to be the same. Let say we have
        # for layers with N_1, ..., N_4 parameters each. Let say after some
        # iterations probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. See below.
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, weight in zip(names, weights):
            n_param = np.prod(weight.shape)
            n_zeros = int(n_param * (cfg.amount))
            n_ones = int(n_param * (1 - cfg.amount))

            if name in _dense_layers:
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros

            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                rhs += n_ones

                if is_kernel:
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (
                                                      np.sum(weight.shape) / np.prod(weight.shape)
                                              ) ** _erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                else:
                    # Cin and Cout for a conv kernel
                    n_in, n_out = weight.shape[:2]
                    raw_probabilities[name] = (n_in + n_out) / (n_in * n_out)
                divisor += raw_probabilities[name] * n_param
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        epsilon = rhs / divisor
        # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    logging.info(f"Density of layer:{mask_name} set to 1.0")
                    _dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    prob_dict = {}
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for name, module in model.named_modules():
        if name in cfg.exclude_layers or name not in names:
            continue
        if name in _dense_layers:
            prob = 1.0
        else:
            prob = epsilon * raw_probabilities[name]

        prob_dict[name] = prob

    return prob_dict
