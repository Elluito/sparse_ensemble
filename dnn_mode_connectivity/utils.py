import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils.prune as prune
import typing
import curves


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.cuda()
        target = target.cuda()

        output = model(input, **kwargs)
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }



def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda()
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]
def remove_reparametrization(model, name_module="", exclude_layer_list: list = []):
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layer_list:
            if name_module == "":
                prune.remove(m, "weight")
            if name == name_module:
                prune.remove(m, "weight")
                break
def weights_to_prune(model: torch.nn.Module, exclude_layer_list=[]):
    modules = []
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layer_list:
            modules.append((m, "weight"))
    return modules
def is_prunable_module(m: torch.nn.Module):
    return (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d))

def prune_with_rate(net: torch.nn.Module, amount: typing.Union[int, float], pruner: str = "erk",
                    type: str = "global",
                    criterion:
                    str =
                    "l1", exclude_layers: list = [], pr_per_layer: dict = {}, return_pr_per_layer: bool = False):
    if type == "global":
        weights = weights_to_prune(net, exclude_layer_list=exclude_layers)
        if criterion == "l1":
            prune.global_unstructured(
                weights,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        if criterion == "l2":
            prune.global_unstructured(
                weights,
                pruning_method=prune.LnStructured,
                amount=amount,
                n=2
            )
    elif type == "layer-wise":
        from layer_adaptive_sparsity.tools.pruners import get_modules, get_weights, weight_pruner_loader
        if pruner == "lamp":
            pruner = weight_pruner_loader(pruner)
            if return_pr_per_layer:
                return pruner(model=net, amount=amount, exclude_layers=exclude_layers,
                              return_amounts=return_pr_per_layer)
            else:
                pruner(model=net, amount=amount, exclude_layers=exclude_layers)
        if pruner == "erk":
            pruner = weight_pruner_loader(pruner)
            pruner(model=net, amount=amount, exclude_layers=exclude_layers)
            # _, amount_per_layer, _, _ = erdos_renyi_per_layer_pruning_rate(model=net, cfg=cfg)
            # names, weights = zip(*get_layer_dict(net))
            # for name, module in net.named_modules():
            #     if name in exclude_layers or name not in names:
            #         continue
            #     else:
            #         prune.l1_unstructured(module, name="weight", amount=float(amount_per_layer[name]))
        if pruner == "manual":
            for name, module in net.named_modules():
                with torch.no_grad():
                    if name in exclude_layers or not is_prunable_module(module):
                        continue
                    else:
                        prune.l1_unstructured(module, name="weight", amount=float(pr_per_layer[name]))



def prepare_check_point_dict(checkpoint:dict,architecture:str,dataset:str):
    from models.alternate_resnet import ResNet18,ResNet50
    from models.alternate_vgg import VGG

    if "model_state" in checkpoint.keys():
        return checkpoint
    else:
        state_dict=None
        if architecture == "ResNet18":
            layers_to_exclude = ["conv1","linear"]
            if dataset == "CIFAR10":
                temp_model = ResNet18(num_classes=10)
                if temp_model.state_dict().keys()<=checkpoint.keys():
                    temp_model.load_state_dict(checkpoint)
                else:
                    prune_with_rate(temp_model,0.9,exclude_layers=layers_to_exclude)
                    temp_model.load_state_dict(checkpoint)
                    remove_reparametrization(temp_model,exclude_layer_list=layers_to_exclude)

                state_dict=temp_model.state_dict()

            if dataset == "CIFAR100":
                temp_model = ResNet18(num_classes=100)
                if temp_model.state_dict().keys()<=checkpoint.keys():
                    temp_model.load_state_dict(checkpoint)
                else:
                    prune_with_rate(temp_model,0.9,exclude_layers=layers_to_exclude)
                    temp_model.load_state_dict(checkpoint)
                    remove_reparametrization(temp_model,exclude_layer_list=layers_to_exclude)
                # remove_reparametrization(temp_model,exclude_layer_list=layers_to_exclude)
                state_dict=temp_model.state_dict()
        if architecture == "VGG19":
            layers_to_exclude = ["features.0","classifier"]
            if dataset == "CIFAR10":
                temp_model = VGG("19",num_classes=10)
                if temp_model.state_dict().keys()<=checkpoint.keys():
                    temp_model.load_state_dict(checkpoint)
                else:
                    prune_with_rate(temp_model,0.9,exclude_layers=layers_to_exclude)
                    temp_model.load_state_dict(checkpoint)
                    remove_reparametrization(temp_model,exclude_layer_list=layers_to_exclude)
                # remove_reparametrization(temp_model,exclude_layer_list=layers_to_exclude)
                state_dict=temp_model.state_dict()
            if dataset == "CIFAR100":
                temp_model = VGG("19",num_classes=100)
                if temp_model.state_dict().keys()<=checkpoint.keys():
                    temp_model.load_state_dict(checkpoint)
                else:
                    prune_with_rate(temp_model,0.9,exclude_layers=layers_to_exclude)
                    temp_model.load_state_dict(checkpoint)
                    remove_reparametrization(temp_model,exclude_layer_list=layers_to_exclude)
                # remove_reparametrization(temp_model,exclude_layer_list=layers_to_exclude)
                state_dict=temp_model.state_dict()

        # best_model_state_dict = torch.load()
        # torch.save({"model_state":best_model.state_dict()},f"noisy_models/{cfg.dataset}/{cfg.architecture}/one_shot_{cfg.pruner}_s{cfg.sigma}_pr{cfg.amount}.pth")
        return {"model_state": state_dict}
def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda()
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))
