import copy
import logging
import time
import typing
import omegaconf
# from main import get_layer_dict
import optuna.samplers
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch import nn
from shrinkbench.metrics.flops import flops
import torch.nn.functional as F
from sklearn.manifold import MDS
from torchmetrics import Accuracy
import wandb
from decimal import Decimal
from flowandprune.imp_estimator import cal_grad
from torch.nn.utils import vector_to_parameters, parameters_to_vector
# from plot_stochastic_pruning import calculate_single_value_from_variance_df
import os
import sys

# from accelerate import Accelerator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: {}".format(device))


########################################################################################################################
######################### PROGRESS BAR FUNCTION ########################################################################
# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
# TOTAL_BAR_LENGTH = 65.
# last_time = time.time()
# begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


############ This function is for projecting a model given a list of models and project them into 2D plane with MDS ####
def project_models(list_of_models: typing.List[torch.nn.Module]):
    dataset = None
    for m in list_of_models:
        vector = parameters_to_vector(m.parameters()).detach().cpu().numpy()
        if dataset is None:
            dataset = np.reshape(vector, (1, -1))
        else:
            dataset = np.vstack((dataset, np.reshape(vector, (1, -1))))
    embeding = MDS(n_components=2)

    transformed_parameters = embeding.fit_transform(dataset)
    return transformed_parameters
    # markers =[".","o","v","^",]


########################################################################################################################
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
    print("Just before creating gbase ")
    gbase = [torch.zeros(p.size()).to(device) for p in base_params]
    print("Just after creating gbase ")

    for inputs, targets in trainloader:
        # print("Just inside of the training loop with data: {}".format(num_data))
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


def test_with_accelerator(net, testloader, one_batch=False, verbose=2, count_flops=False, batch_flops=0,
                          accelerator=None):
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    if count_flops:
        assert batch_flops != 0, "If count_flops is True, batch_flops must be non-zero"
    sparse_flops = 0
    first_time = 1
    sparse_flops_batch = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            all_predictions, all_targets = accelerator.gather_for_metrics((outputs, targets))

            if count_flops:
                sparse_flops += batch_flops
            test_loss += loss.data.item()
            _, predicted = torch.max(all_predictions.data, 1)
            total += targets.size(0)
            correct += predicted.eq(all_targets.data).cpu().sum()

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
    if count_flops:
        return 100. * correct.item() / total, sparse_flops
    else:
        return 100. * correct.item() / total


def train(epoch, net, trainloader, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.to(device)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print("Batch accuracy: {}".format(100. * correct / total, correct, total))


def test(net, use_cuda, testloader, one_batch=False, verbose=2, count_flops=False, batch_flops=0, number_batches=0):
    if use_cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    if count_flops:
        assert batch_flops != 0, "If count_flops is True,batch_flops must be non-zero"

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
            if torch.all(outputs > 0):
                _, predicted = torch.max(outputs.data, 1)
            else:
                soft_max_outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(soft_max_outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            # print(correct/total)

            if batch_idx % 10 == 0:
                if verbose == 2:
                    print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                          % (test_loss / (batch_idx + 1), 100. * correct.item() / total, correct, total))
                    print("Batch first input")
                    print("{}".format(inputs[0].cpu().numpy()))
                    print("Output of the model")
                    print("{}".format(outputs[0].cpu().numpy()))
                    print("Predicted")
                    print("{}".format(predicted[0].cpu().numpy()))
                    print("targets")
                    print("{}".format(targets[0].cpu().numpy()))
            if one_batch:
                if count_flops:
                    return 100. * correct.item() / total, sparse_flops
                else:
                    return 100. * correct.item() / total

            if number_batches > 0:
                if number_batches < batch_idx:
                    return 100. * correct.item() / total

    if verbose == 1 or verbose == 2:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss / len(testloader), correct, total,
            100. * correct.item() / total))
    # net.cpu()
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


# COMMENT: THIS IS FOR THE EXPERIMENTS OF FINE-TUNING UNRESTRICTED AN then compare the final solution to the deterministic to see if they are on a different basin still.
def unrestricted_fine_tune_measure_flops(pruned_model: nn.Module, dataLoader: torch.utils.data.DataLoader,
                                         testLoader: torch.utils.data.DataLoader,
                                         epochs=1,
                                         FLOP_limit: float = 0, initial_flops=0, use_wandb=False, exclude_layers=[],
                                         fine_tune_exclude_layers=False, fine_tune_non_zero_weights=True,
                                         gradient_flow_file_prefix="", cfg=None, ):
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

    # TODO: Here I need to be carefull of how I do the recording since this model is unrestricted
    data, y = next(iter(dataLoader))
    forward_pass_dense_flops, forward_pass_sparse_flops = flops(pruned_model, data)

    file_path = None
    weights_path = ""
    if gradient_flow_file_prefix != "":
        file_path = gradient_flow_file_prefix
        file_path += "recordings.csv"

        if Path(gradient_flow_file_prefix).owner() == "sclaam":
            weights_file_path = "/nobackup/sclaam/" + gradient_flow_file_prefix + "weigths/"
        if Path(gradient_flow_file_prefix).owner() == "luisaam":
            weights_file_path = "GF_data/" + gradient_flow_file_prefix + "weigths/"

        weights_path = Path(weights_file_path)
        weights_path.mkdir(parents=True)
        measure_and_record_gradient_flow(pruned_model, dataLoader, testLoader, cfg, file_path,
                                         total_FLOPS + forward_pass_sparse_flops + 2 * forward_pass_sparse_flops, -1,
                                         mask_dict=mask_dict, use_wandb=use_wandb)

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

        if epoch % 10 == 0 and gradient_flow_file_prefix != "":
            measure_and_record_gradient_flow(pruned_model, dataLoader, testLoader, cfg, file_path, total_FLOPS, epoch,
                                             mask_dict=mask_dict
                                             , use_wandb=use_wandb)
            state_dict = pruned_model.state_dict()
            temp_name = weights_path / "epoch_{}.pth".format(epoch)
            torch.save(state_dict, temp_name)
        if FLOP_limit != 0:
            if total_FLOPS > FLOP_limit:
                break
    if gradient_flow_file_prefix != "":
        measure_and_record_gradient_flow(pruned_model, dataLoader, testLoader, cfg, file_path, total_FLOPS, epochs,
                                         mask_dict=mask_dict
                                         , use_wandb=use_wandb)

    test_set_performance = test(pruned_model, use_cuda=True, testloader=testLoader)

    if use_wandb:
        if gradient_flow_file_prefix != "":
            df = pd.read_csv(file_path, sep=",", header=0, index_col=False)
            table = wandb.Table(data=df)
            wandb.log({"Gradient Flow results": table})
        wandb.log({
            "test_set_accuracy": test_set_performance,
            "dense_flops": total_FLOPS,
            "final_accuracy": test_set_performance
        })

    return total_FLOPS
    # msg_perormance = f"{performance:.2f}".replace(".", ",")


def restricted_IMAGENET_fine_tune_ACCELERATOR_measure_flops(pruned_model: nn.Module,
                                                            dataLoader_p: torch.utils.data.DataLoader,
                                                            testLoader_p: torch.utils.data.DataLoader,
                                                            epochs=1,
                                                            FLOP_limit: float = 0, initial_flops=0, use_wandb=False,
                                                            exclude_layers=[],
                                                            fine_tune_exclude_layers=False,
                                                            fine_tune_non_zero_weights=True,
                                                            gradient_flow_file_prefix="", cfg=None):
    disable_bn(pruned_model)

    optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.0001,
                                momentum=0.9, weight_decay=5e-4)
    if "cifar" in cfg.dataset:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    grad_clip = 0
    if cfg.gradient_cliping:
        grad_clip = 0.1
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

    # apply_mask_with_hook(pruned_model, mask_dict)
    if not fine_tune_exclude_layers:
        disable_exclude_layers(pruned_model, exclude_layers)
    if not fine_tune_non_zero_weights:
        disable_all_except(pruned_model, exclude_layers)
    ######################## Prepare with the accelerator##############################
    accelerator = Accelerator(mixed_precision="fp16")
    pruned_model, optimizer, dataLoader, testLoader, lr_scheduler = accelerator.prepare(pruned_model, optimizer,
                                                                                        dataLoader_p, testLoader_p,
                                                                                        lr_scheduler)
    total_FLOPS = 0
    total_sparse_FLOPS = initial_flops
    # first_time = 1

    data, y = next(iter(dataLoader))
    forward_pass_dense_flops, forward_pass_sparse_flops = flops(pruned_model, data)
    ######################### for record and saving stuf##############################
    file_path = None
    weights_path = ""
    test_accuracy = -1
    if gradient_flow_file_prefix != "":
        file_path = gradient_flow_file_prefix
        file_path += "recordings.csv"

        if Path(gradient_flow_file_prefix).owner() == "sclaam":
            weights_file_path = "/nobackup/sclaam/" + gradient_flow_file_prefix + "weigths/"
        if Path(gradient_flow_file_prefix).owner() == "luisaam":
            weights_file_path = "GF_data/" + gradient_flow_file_prefix + "weigths/"

        weights_path = Path(weights_file_path)
        weights_path.mkdir(parents=True)
        test_accuracy, val_accuracy = measure_and_record_gradient_flow(accelerator.unwrap_model(pruned_model),
                                                                       dataLoader_p, testLoader_p, cfg,
                                                                       file_path,
                                                                       total_sparse_FLOPS, -1,
                                                                       use_wandb=use_wandb, mask_dict=mask_dict)

        # state_dict = pruned_model.state_dict()
        # temp_name = weights_path / "epoch_{}.pth".format(-1)
        # torch.save(state_dict,temp_name)

    # pruned_model.cuda()
    # pruned_model.train()
    #################### for knowing when to save  the model###################
    best_accuracy = -1

    criterion = nn.CrossEntropyLoss()
    average_accuracy = AverageMeter()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataLoader):

            # data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # first forward-backward step
            predictions = pruned_model(data)
            # enable_bn(model)
            loss = criterion(predictions, target)

            # loss calculation
            accelerator.backward(loss)

            backward_flops_sparse = 2 * forward_pass_sparse_flops
            backward_flops_dense = 2 * forward_pass_dense_flops
            batch_dense_flops = forward_pass_dense_flops + backward_flops_dense
            batch_sparse_flops = forward_pass_sparse_flops + backward_flops_sparse
            total_FLOPS += batch_dense_flops
            total_sparse_FLOPS += batch_sparse_flops
            accuracy.update(preds=predictions, target=target)
            # Mask the grad_
            mask_gradient(pruned_model, mask_dict=mask_dict)

            if cfg.gradient_cliping:
                accelerator.clip_grad_value_(pruned_model.parameters(), grad_clip)

            optimizer.step()

            lr_scheduler.step()

            if batch_idx % 10 == 0 or FLOP_limit != 0:
                acc = accuracy.compute()
                average_accuracy.update(acc)
                flops_sparse = '%.3E' % Decimal(total_sparse_FLOPS)
                print(f"Fine-tune Results - Epoch: {epoch}  Avg accuracy: {acc:.2f} Avg loss:"
                      f" {loss.item():.2f} FLOPS:{flops_sparse} sparsity {sparsity(pruned_model) :.3f}")

                # W&B Logging
                if use_wandb:
                    acc = accuracy.compute()
                    # test_accuracy = test(accelerator.unwrap_model(pruned_model), use_cuda=True, testloader=[get_random_batch(testLoader)],
                    #                      one_batch=True,verbose=0)
                    wandb.log({
                        "val_set_accuracy": acc * 100,
                        "sparse_flops": total_sparse_FLOPS,
                        # "test_set_accuracy": test_accuracy,
                        "sparsity": sparsity(pruned_model)
                    })
                if FLOP_limit != 0 and FLOP_limit > total_sparse_FLOPS:
                    break

        # if gradient_flow_file_prefix != "":
        # After every epoch, save the model
        if best_accuracy < test_accuracy:
            unwraped_model = accelerator.unwrap_model(pruned_model)
            state_dict = unwraped_model.state_dict()
            save = {"net": state_dict, "test_accuracy": average_accuracy.avg, "epoch": epoch}
            temp_name = weights_path / "net.pth"
            torch.save(save, temp_name)
            best_accuracy = test_accuracy
            del unwraped_model
        if epoch % 5 == 0 and gradient_flow_file_prefix != "":

            unwraped_model = accelerator.unwrap_model(pruned_model)
            test_accuracy, val_accuracy = measure_and_record_gradient_flow(unwraped_model, dataLoader_p, testLoader_p,
                                                                           cfg, file_path,
                                                                           total_sparse_FLOPS, epoch,
                                                                           use_wandb=use_wandb, mask_dict=mask_dict)
            if best_accuracy < test_accuracy:
                state_dict = unwraped_model.state_dict()
                save = {"net": state_dict, "test_accuracy": average_accuracy.avg, "epoch": epoch}
                temp_name = weights_path / "net.pth"
                torch.save(save, temp_name)
                best_accuracy = test_accuracy
            if use_wandb:
                acc = accuracy.compute()
                # test_accuracy = test(accelerator.unwrap_model(pruned_model), use_cuda=True, testloader=[get_random_batch(testLoader)],
                #                      one_batch=True,verbose=0)
                wandb.log({
                    "val_set_accuracy": acc * 100,
                    "sparse_flops": total_sparse_FLOPS,
                    "test_set_accuracy": test_accuracy,
                    "sparsity": sparsity(pruned_model)
                })
            del unwraped_model
            # state_dict = pruned_model.state_dict()
            # temp_name = weights_path / "epoch_{}.pth".format(epoch)
            # torch.save(state_dict,temp_name)
        if FLOP_limit != 0:
            if total_sparse_FLOPS > FLOP_limit:
                break
        ############ Reset the average counter if  we are not in the last epoch ######################

        if epoch < epochs - 1:
            average_accuracy.reset()

    ################################################## Outside of epochs loop ###################

    if gradient_flow_file_prefix != "":
        unwraped_model = accelerator.unwrap_model(pruned_model)
        test_accuracy, val_accuracy = measure_and_record_gradient_flow(unwraped_model, dataLoader_p, testLoader_p, cfg,
                                                                       file_path,
                                                                       total_sparse_FLOPS, epochs,
                                                                       use_wandb=use_wandb, mask_dict=mask_dict)

        if best_accuracy < test_accuracy:
            state_dict = unwraped_model.state_dict()
            save = {"net": state_dict, "test_accuracy": average_accuracy.avg, "epoch": epochs}
            temp_name = weights_path / "net.pth"
            torch.save(save, temp_name)
        del unwraped_model
    if use_wandb:

        test_set_performance = test_with_accelerator(pruned_model, testloader=testLoader)
        if gradient_flow_file_prefix != "":
            df = pd.read_csv(file_path, sep=",", header=0, index_col=False)
            table = wandb.Table(data=df)
            wandb.log({"Gradient Flow results": table})
        wandb.log({
            "test_set_accuracy": test_set_performance,
            "sparse_flops": total_sparse_FLOPS,
            "final_accuracy": test_set_performance
        })

    return total_sparse_FLOPS


def restricted_fine_tune_measure_flops_sto_and_deterministic(pruned_model: nn.Module,deterministic_pruned_model: nn.Module, dataLoader: torch.utils.data.DataLoader,
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
    optimizer2 = torch.optim.SGD(deterministic_pruned_model.parameters(), lr=0.0001,
                                momentum=0.9, weight_decay=5e-4)

    if "cifar" in cfg.dataset:
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) ===> original code before 21/01/2025
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # ===> code after 21/01/2025
        lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs)  # ===> code after 21/01/2025

    else:

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) original code before 21/01/2025
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # code after 21/01/2025
        lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs)  # code after 21/01/2025

    grad_clip = 0
    if cfg.gradient_cliping:
        grad_clip = 0.1
    names, weights = zip(*get_layer_dict(pruned_model))
    if cfg.dataset == "cifar10" or cfg.dataset == "mnist":
        accuracy = Accuracy(task="multiclass", num_classes=10).to("cuda")
        accuracy2 = Accuracy(task="multiclass", num_classes=10).to("cuda")
    if cfg.dataset == "cifar100":
        accuracy = Accuracy(task="multiclass", num_classes=100).to("cuda")
        accuracy2 = Accuracy(task="multiclass", num_classes=100).to("cuda")
    if cfg.dataset == "imagenet":
        accuracy = Accuracy(task="multiclass", num_classes=1000).to("cuda")

    mask_dict = get_mask(model=pruned_model)
    mask_dict2 = get_mask(model=deterministic_pruned_model)
    for name in exclude_layers:
        if name in list(mask_dict.keys()):
            mask_dict.pop(name)
            mask_dict2.pop(name)
    total_FLOPS = 0
    total_sparse_FLOPS = initial_flops
    # first_time = 1

    data, y = next(iter(dataLoader))
    data = data.cuda()
    # forward_pass_dense_flops, forward_pass_sparse_flops = flops(pruned_model, data)
    #
    # file_path = None
    # weights_path = ""
    # if gradient_flow_file_prefix != "":
    #
    #     file_path = gradient_flow_file_prefix
    #     file_path += "recordings.csv"
    #
    #     if Path(gradient_flow_file_prefix).owner() == "sclaam":
    #         weights_file_path = "/nobackup/sclaam/" + gradient_flow_file_prefix + "weigths/"
    #     if Path(gradient_flow_file_prefix).owner() == "luisaam":
    #         weights_file_path = "GF_data/" + gradient_flow_file_prefix + "weigths/"
    #
    #     weights_path = Path(weights_file_path)
    #     weights_path.mkdir(parents=True)
    #     measure_and_record_gradient_flow(pruned_model, dataLoader, testLoader, cfg, file_path, total_sparse_FLOPS, -1,
    #                                      mask_dict=mask_dict, use_wandb=use_wandb)
    #     state_dict = pruned_model.state_dict()
    #     temp_name = weights_path / "epoch_OS.pth"
    #     torch.save(state_dict,temp_name)

    deterministic_pruned_model.cuda()
    deterministic_pruned_model.train()
    pruned_model.cuda()
    pruned_model.train()
    disable_bn(pruned_model)
    if not fine_tune_exclude_layers:
        disable_exclude_layers(pruned_model, exclude_layers)
        disable_exclude_layers(deterministic_pruned_model, exclude_layers)
    if not fine_tune_non_zero_weights:
        disable_all_except(pruned_model, exclude_layers)
        disable_all_except(deterministic_pruned_model, exclude_layers)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataLoader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            optimizer2.zero_grad()
            # first forward-backward step
            predictions = pruned_model(data)
            predictions2 =deterministic_pruned_model(data)
            # enable_bn(model)
            loss = criterion(predictions, target)
            loss2 = criterion(predictions2, target)

            loss.backward()
            loss2.backward()
            # backward_flops_sparse = 2 * forward_pass_sparse_flops
            # backward_flops_dense = 2 * forward_pass_dense_flops
            # batch_dense_flops = forward_pass_dense_flops + backward_flops_dense
            # batch_sparse_flops = forward_pass_sparse_flops + backward_flops_sparse
            # total_FLOPS += batch_dense_flops
            # total_sparse_FLOPS += batch_sparse_flops

            accuracy.update(preds=predictions.cuda(), target=target.cuda())
            accuracy.update(preds=predictions2.cuda(), target=target.cuda())

            # Mask the grad_
            mask_gradient(pruned_model, mask_dict=mask_dict)
            mask_gradient(deterministic_pruned_model, mask_dict=mask_dict2)

            if grad_clip:
                nn.utils.clip_grad_value_(pruned_model.parameters(), grad_clip)
                nn.utils.clip_grad_value_(deterministic_pruned_model.parameters(), grad_clip)

            optimizer.step()
            optimizer2.step()

            lr_scheduler.step()
            lr_scheduler2.step()

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
                # acc = accuracy.compute()
                # flops_sparse = '%.3E' % Decimal(total_sparse_FLOPS)
                # print(f"Fine-tune sotch Results - Epoch: {epoch}  Avg accuracy: {acc:.2f} Avg loss:"
                #       f" {loss.item():.2f} FLOPS:{flops_sparse} sparsity {sparsity(pruned_model) :.3f}")
                # acc2 = accuracy2.compute()
                # print(f"Fine-tune Results - Epoch: {epoch}  Avg accuracy: {acc:.2f} Avg loss:"
                #       f" {loss.item():.2f} FLOPS:{flops_sparse} sparsity {sparsity(pruned_model) :.3f}")
                pass

                if FLOP_limit != 0 and FLOP_limit > total_sparse_FLOPS:
                    break
        # if gradient_flow_file_prefix != "":

        # if epoch % 10 == 0 and gradient_flow_file_prefix != "":
        #     measure_and_record_gradient_flow(pruned_model, dataLoader, testLoader, cfg, file_path, total_sparse_FLOPS,
        #                                      epoch,
        #                                      mask_dict=mask_dict
        #                                      , use_wandb=use_wandb)

            # state_dict = pruned_model.state_dict()
            # temp_name = weights_path / "epoch_{}.pth".format(epoch)
            # torch.save(state_dict,temp_name)
        #
        # if FLOP_limit != 0:
        #     if total_sparse_FLOPS > FLOP_limit:
        #         break
    # if gradient_flow_file_prefix != "":
    #     measure_and_record_gradient_flow(pruned_model, dataLoader, testLoader, cfg, file_path, total_sparse_FLOPS,
    #                                      epochs,
    #                                      mask_dict=mask_dict
    #                                      , use_wandb=use_wandb)
    #     state_dict = pruned_model.state_dict()
    #     temp_name = weights_path / "epoch_{}.pth".format(epochs - 1)
    #     torch.save(state_dict, temp_name)

    test_set_performance = test(pruned_model, use_cuda=True, testloader=testLoader)
    test_set_performance2 = test(deterministic_pruned_model, use_cuda=True, testloader=testLoader)
    difference_in_performance =test_set_performance-test_set_performance2

    # if not os.path.isdir(save_folder):
    #     os.mkdir(save_folder)
    # if os.path.isfile('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc)):
    #     os.remove('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc))
    # torch.save(state, '{}/{}_test_acc_{}.pth'.format(save_folder, name, acc))
    # best_acc = acc

    # if use_wandb:
    #     if gradient_flow_file_prefix != "":
    #         df = pd.read_csv(file_path, sep=",", header=0, index_col=False)
    #         table = wandb.Table(data=df)
    #         wandb.log({"Gradient Flow results": table})
    #     wandb.log({
    #         "test_set_accuracy": test_set_performance,
    #         "sparse_flops": total_sparse_FLOPS,
    #         "final_accuracy": test_set_performance
    #     })

    return test_set_performance,difference_in_performance

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
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) ===> original code before 21/01/2025
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # ===> code after 21/01/2025

    else:

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) original code before 21/01/2025
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # code after 21/01/2025

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
    data = data.cuda()
    forward_pass_dense_flops, forward_pass_sparse_flops = flops(pruned_model, data)
    # forward_pass_dense_flops, forward_pass_sparse_flops = 0,0

    file_path = None
    weights_path = ""
    if gradient_flow_file_prefix != "":

        file_path = gradient_flow_file_prefix
        file_path += "recordings.csv"

        if Path(gradient_flow_file_prefix).owner() == "sclaam":
            # weights_file_path = "/nobackup/sclaam/" + gradient_flow_file_prefix + "weigths/"
            weights_file_path = "/mnt/scratch/sclaam/" + gradient_flow_file_prefix + "weights/"
        if Path(gradient_flow_file_prefix).owner() == "luisaam":
            weights_file_path = "GF_data/" + gradient_flow_file_prefix + "weigths/"

        weights_path = Path(weights_file_path)
        weights_path.mkdir(parents=True,exist_ok=True)
        measure_and_record_gradient_flow(pruned_model, dataLoader, testLoader, cfg, file_path, total_sparse_FLOPS, -1,
                                         mask_dict=mask_dict, use_wandb=use_wandb)
        state_dict = pruned_model.state_dict()
        temp_name = weights_path / "epoch_OS.pth"
        torch.save(state_dict,temp_name)

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

        if epoch % 10 == 0 and gradient_flow_file_prefix != "":
            measure_and_record_gradient_flow(pruned_model, dataLoader, testLoader, cfg, file_path, total_sparse_FLOPS,
                                             epoch,
                                             mask_dict=mask_dict
                                             , use_wandb=use_wandb)

            # state_dict = pruned_model.state_dict()
            # temp_name = weights_path / "epoch_{}.pth".format(epoch)
            # torch.save(state_dict,temp_name)

        if FLOP_limit != 0:
            if total_sparse_FLOPS > FLOP_limit:
                break
    if gradient_flow_file_prefix != "":
        measure_and_record_gradient_flow(pruned_model, dataLoader, testLoader, cfg, file_path, total_sparse_FLOPS,
                                         epochs,
                                         mask_dict=mask_dict
                                         , use_wandb=use_wandb)
        state_dict = pruned_model.state_dict()
        temp_name = weights_path / "epoch_{}.pth".format(epochs - 1)
        torch.save(state_dict, temp_name)

    test_set_performance = test(pruned_model, use_cuda=True, testloader=testLoader)

    # if not os.path.isdir(save_folder):
    #     os.mkdir(save_folder)
    # if os.path.isfile('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc)):
    #     os.remove('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc))
    # torch.save(state, '{}/{}_test_acc_{}.pth'.format(save_folder, name, acc))
    # best_acc = acc

    if use_wandb:
        if gradient_flow_file_prefix != "":
            df = pd.read_csv(file_path, sep=",", header=0, index_col=False)
            table = wandb.Table(data=df)
            wandb.log({"Gradient Flow results": table})
        wandb.log({
            "test_set_accuracy": test_set_performance,
            "sparse_flops": total_sparse_FLOPS,
            "final_accuracy": test_set_performance
        })

    return total_sparse_FLOPS,pruned_model


def restricted_fine_tune_measure_flops_calc_variance(pruned_model: nn.Module, dataLoader: torch.utils.data.DataLoader,
                                                     testLoader: torch.utils.data.DataLoader,
                                                     epochs=1,
                                                     FLOP_limit: float = 0, initial_flops=0, use_wandb=False,
                                                     exclude_layers=[],
                                                     fine_tune_exclude_layers=False, fine_tune_non_zero_weights=True,
                                                     gradient_flow_file_prefix="", cfg=None):
    # optimizer = torch.optim.SGD()
    #################### Best accuracy yet ################################

    ####################
    optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.0001,
                                momentum=0.9, weight_decay=5e-4)
    if "cifar" in cfg.dataset:
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) ===> original code before 21/01/2025
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # ===> code after 21/01/2025

    else:

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) original code before 21/01/2025
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # code after 21/01/2025

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
    data = data.cuda()
    forward_pass_dense_flops, forward_pass_sparse_flops = flops(pruned_model, data)

    file_path = None
    weights_path = ""
    if gradient_flow_file_prefix != "":

        file_path = gradient_flow_file_prefix
        file_path += "recordings.csv"

        if Path(gradient_flow_file_prefix).owner() == "sclaam":
            weights_file_path = "/nobackup/sclaam/" + gradient_flow_file_prefix + "weigths/"
        if Path(gradient_flow_file_prefix).owner() == "luisaam":
            weights_file_path = "GF_data/" + gradient_flow_file_prefix + "weigths/"

        weights_path = Path(weights_file_path)
        weights_path.mkdir(parents=True)
        measure_and_record_gradient_flow(pruned_model, dataLoader, testLoader, cfg, file_path, total_sparse_FLOPS, -1,
                                         mask_dict=mask_dict, use_wandb=use_wandb)
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

        if epoch % 10 == 0 and gradient_flow_file_prefix != "":
            measure_and_record_gradient_flow(pruned_model, dataLoader, testLoader, cfg, file_path, total_sparse_FLOPS,
                                             epoch,
                                             mask_dict=mask_dict
                                             , use_wandb=use_wandb)

            # state_dict = pruned_model.state_dict()
            # temp_name = weights_path / "epoch_{}.pth".format(epoch)
            # torch.save(state_dict,temp_name)

        if FLOP_limit != 0:
            if total_sparse_FLOPS > FLOP_limit:
                break
    if gradient_flow_file_prefix != "":
        measure_and_record_gradient_flow(pruned_model, dataLoader, testLoader, cfg, file_path, total_sparse_FLOPS,
                                         epochs,
                                         mask_dict=mask_dict
                                         , use_wandb=use_wandb)
        state_dict = pruned_model.state_dict()
        temp_name = weights_path / "epoch_{}.pth".format(epochs - 1)
        torch.save(state_dict, temp_name)

    test_set_performance = test(pruned_model, use_cuda=True, testloader=testLoader)

    # if not os.path.isdir(save_folder):
    #     os.mkdir(save_folder)
    # if os.path.isfile('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc)):
    #     os.remove('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc))
    # torch.save(state, '{}/{}_test_acc_{}.pth'.format(save_folder, name, acc))
    # best_acc = acc

    if use_wandb:
        if gradient_flow_file_prefix != "":
            df = pd.read_csv(file_path, sep=",", header=0, index_col=False)
            table = wandb.Table(data=df)
            wandb.log({"Gradient Flow results": table})
        wandb.log({
            "test_set_accuracy": test_set_performance,
            "sparse_flops": total_sparse_FLOPS,
            "final_accuracy": test_set_performance
        })

    return total_sparse_FLOPS


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
    #
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
def apply_mask(model: nn.Module, mask_dict: dict):
    for name, module in model.named_modules():
        if name in mask_dict.keys():
            module.weight.data.mul_(mask_dict[name])


@torch.no_grad()
def apply_mask_with_hook(model: nn.Module, mask_dict: dict):
    '''

    @param model: model to mask
    @param mask_dict:dict of module names and mask
    @return:
    '''
    for name, module in model.named_modules():
        if name in mask_dict.keys():
            def hook(module_p, grad_input, grad_output) -> tuple[torch.Tensor] or None:
                module_p.weight.data.mul_(module_p.mask)

            module.register_buffer("mask", mask_dict[name])
            module.register_full_backward_hook(hook)


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


def measure_and_record_gradient_flow_with_ACCELERATOR(wrapped_model: nn.Module, accelerator,
                                                      dataLoader, testLoader, filepath, total_flops, epoch,
                                                      use_wandb=False, criterion=None):
    # model = accelerator.unwrap_model(wrapped_model)

    print("Just before cal_grad call")
    # Calculate everything with respect to the validation set
    val_dict = {}
    t0 = time.time()
    grad: typing.List[torch.Tensor] = cal_grad_ACCELERATOR(wrapped_model, trainloader=dataLoader,
                                                           device=torch.device("cuda"), criterion=criterion)
    t1 = time.time()
    print("Gradient calculation on val-set with unwrapped model {}".format(t1 - t0))
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
    t0 = time.time()
    accuracy = test_with_accelerator(wrapped_model, dataLoader, verbose=0)
    t1 = time.time()
    print("Evaluation on val-set with wrapped model {}".format(t1 - t0))

    # print("Calculating eigenvalues on validation set for epoch:{}".format(epoch))

    # top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=5,maxIter=20)
    # print("Calculating hessian trace on validation set for epoch:{}".format(epoch))
    # trace = hessian_comp.trace(maxIter=20)
    # for i,value in enumerate(top_eigenvalues):
    #     val_dict["val_set_EV{}".format(i)] = [value]
    # val_dict["val_set_trace"] = [trace]
    # # density_eigen, density_weight = hessian_comp.density()
    val_dict["val_accuracy"] = [accuracy]

    # Calculate everything with respect to the test set
    test_dict = {}
    grad: typing.List[torch.Tensor] = cal_grad_ACCELERATOR(wrapped_model, trainloader=testLoader,
                                                           device=torch.device("cuda"), criterion=criterion)

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
    accuracy = test_with_accelerator(wrapped_model, testLoader, verbose=0)
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
        for n, v in val_dict.items():
            log_dict[n] = v[0]
        for n, v in test_dict.items():
            log_dict[n] = v[0]
        log_dict.update(val_dict)
        log_dict.update(test_dict)
        wandb.log(log_dict)

    return accuracy


def measuring_feature_sample_variance_from_model(net, evaluation_set, cfg, name: str = ""):
    use_cuda = torch.cuda.is_available()
    N = 5
    pop = []
    pruned_performance = []
    stochastic_dense_performances = []
    stochastic_deltas = []

    t0 = time.time()
    original_performance = test(net, use_cuda, evaluation_set, verbose=1)
    t1 = time.time()
    print("Time for test: {}".format(t1 - t0))
    pruned_original = copy.deepcopy(net)

    names, weights = zip(*get_layer_dict(net))
    number_of_layers = len(names)
    sigma_per_layer = dict(zip(names, [cfg.sigma] * number_of_layers))

    if cfg.pruner == "global":
        prune_with_rate(pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")
    else:
        prune_with_rate(pruned_original, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner=cfg.pruner)

    remove_reparametrization(pruned_original, exclude_layer_list=cfg.exclude_layers)

    print("pruned_performance of pruned original")
    t0 = time.time()
    pruned_original_performance = test(pruned_original, use_cuda, evaluation_set, verbose=1)
    print("Det_performance in function: {}".format(pruned_original_performance))
    t1 = time.time()
    print("Time for test: {}".format(t1 - t0))

    deter_original_variance = calculate_variance_models_dataloader(net, evaluation_set, "cuda")
    deter_original_df = pd.DataFrame.from_dict(deter_original_variance)

    # deter_original_df.to_csv(f"variance_collapse/{cfg.model}_{cfg.dataset}_pr_{cfg.amount}_{cfg.pruner}_original_deter_l2_mean.csv",
    #                          sep=",")

    # deter_original_df.to_csv(
    #     f"variance_collapse/{cfg.model}_{cfg.dataset}_pr_{cfg.amount}_{cfg.pruner}_original_dense_deter_var_mean.csv",
    #     sep=",")
    deter_original_variance = calculate_variance_models_dataloader(pruned_original, evaluation_set, "cuda")
    deter_original_pruned_df = pd.DataFrame.from_dict(deter_original_variance)
    # deter_original_pruned_df.to_csv(
    #     f"variance_collapse/{cfg.model}_{cfg.dataset}_pr_{cfg.amount}_{cfg.pruner}_original_pruned_deter_var_mean.csv",
    #     sep=",")

    del pruned_original
    # pop.append(pruned_original)
    # pruned_performance.append(pruned_original_performance)
    labels = []
    # stochastic_dense_performances.append(original_performance)
    all_noisy_models = None
    all_noisy_models_dense = None

    for n in range(N):
        dense_current_model = get_noisy_sample_sigma_per_layer(net, cfg, sigma_per_layer=sigma_per_layer)
        # stochastic_with_deterministic_mask_performance.append(det_mask_transfer_model_performance)
        print("Stochastic dense performance")
        t0 = time.time()
        StoDense_performance = test(dense_current_model, use_cuda, evaluation_set, verbose=1)
        t1 = time.time()
        print("Time for test: {}".format(t1 - t0))
        # Dense stochastic performance
        stochastic_dense_performances.append(StoDense_performance)

        current_model = copy.deepcopy(dense_current_model)

        if cfg.pruner == "global":
            prune_with_rate(current_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="global")
        else:
            prune_with_rate(current_model, cfg.amount, exclude_layers=cfg.exclude_layers, type="layer-wise",
                            pruner=cfg.pruner)

        # Here is where I transfer the mask from the pruned stochastic model to the
        # original weights and put it in the ranking
        # copy_buffers(from_net=current_model, to_net=sto_mask_transfer_model)
        remove_reparametrization(current_model, exclude_layer_list=cfg.exclude_layers)
        # record_predictions(current_model, evaluation_set,
        #                    "{}_one_shot_sto_{}_predictions_{}".format(cfg.architecture, cfg.model_type, cfg.dataset))
        torch.cuda.empty_cache()
        print("Stocastic pruning performance")
        stochastic_pruned_performance = test(current_model, use_cuda, evaluation_set, verbose=1)
        print("Time for test: {}".format(t1 - t0))

        pruned_performance.append(stochastic_pruned_performance)
        stochastic_deltas.append(StoDense_performance - stochastic_pruned_performance)

        sto_noisy_variance = calculate_variance_models_dataloader(current_model, evaluation_set,
                                                                  "cuda")
        sto_noisy_df = pd.DataFrame.from_dict(sto_noisy_variance)
        del current_model
        torch.cuda.empty_cache()
        if all_noisy_models is None:
            all_noisy_models = sto_noisy_df
        else:
            all_noisy_models = pd.concat((all_noisy_models, sto_noisy_df), ignore_index=True)

        sto_noisy_variance_dense = calculate_variance_models_dataloader(dense_current_model, evaluation_set,
                                                                        "cuda")
        sto_noisy_df_dense = pd.DataFrame.from_dict(sto_noisy_variance_dense)
        if all_noisy_models_dense is None:
            all_noisy_models_dense = sto_noisy_df_dense
        else:
            all_noisy_models_dense = pd.concat((all_noisy_models_dense, sto_noisy_df_dense), ignore_index=True)
    # all_noisy_models.to_csv(
    #     f"variance_collapse/{cfg.model}_{cfg.dataset}_noisy_sto_pr_{cfg.amount}_{cfg.pruner}_sigma_{cfg.sigma}_l2_mean.csv", sep=",")

    # all_noisy_models.to_csv(
    #     f"variance_collapse/{cfg.model}_{cfg.dataset}_noisy_sto_pr_{cfg.amount}_{cfg.pruner}_sigma_{cfg.sigma}_pruned_var_mean.csv",
    #     sep=",")
    #
    # all_noisy_models_dense.to_csv(
    #     f"variance_collapse/{cfg.model}_{cfg.dataset}_noisy_sto_pr_{cfg.amount}_{cfg.pruner}_sigma_{cfg.sigma}_dense_var_mean.csv",
    #     sep=",")

    return deter_original_df, deter_original_pruned_df, all_noisy_models, all_noisy_models_dense


def measure_and_record_gradient_flow(model: nn.Module, dataLoader, testLoader, cfg, filepath, total_flops, epoch,
                                     mask_dict, use_wandb=False, record=True, record_variance=False):
    model = copy.deepcopy(model)
    t_begining = time.time()
    disable_bn(model)
    if not cfg.fine_tune_exclude_layers:
        disable_exclude_layers(model, cfg.exclude_layers)
    if not cfg.fine_tune_non_zero_weights:
        disable_all_except(model, cfg.exclude_layers)
    model.to(device=device)

    ###################################################
    # Calculate everything with respect to the validation set
    ###################################################

    val_dict = {}
    print("just before cal_grad")
    t0 = time.time()
    grad: typing.List[torch.Tensor] = cal_grad(model, trainloader=dataLoader)
    print("Just finished cal_grad")
    t1 = time.time()
    print("Time in cal_grad valset: {} s".format(t1 - t0))
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

    accuracy = test(model, True if device == "cuda" else False, dataLoader, verbose=0)
    model.to(device)

    # print("Calculating eigenvalues on validation set for epoch:{}".format(epoch))
    # top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=5,maxIter=20)
    # print("Calculating hessian trace on validation set for epoch:{}".format(epoch))
    # trace = hessian_comp.trace(maxIter=20)
    # for i,value in enumerate(top_eigenvalues):
    #     val_dict["val_set_EV{}".format(i)] = [value]
    # val_dict["val_set_trace"] = [trace]
    # # density_eigen, density_weight = hessian_comp.density()

    val_dict["val_accuracy"] = [accuracy]
    ###################################################
    # Calculate everything with respect to the test set
    ###################################################
    test_dict = {}
    t0 = time.time()
    grad: typing.List[torch.Tensor] = cal_grad(model, trainloader=testLoader)
    # t0 = time.time()
    t1 = time.time()
    print("Time in cal_grad test_set: {} s".format(t1 - t0))
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

    # if record_variance:
    #     deter_original_dense_df, deter_original_pruned_df, all_noisy_models, all_noisy_models_dense = measuring_feature_sample_variance_from_model(
    #         model, cfg, evaluation_set=dataLoader, name="")
    #     clean_variance, noisy_variance = calculate_single_value_from_variance_df(
    #         noisy_variance_dense=all_noisy_models_dense
    #         , clean_variance_dense=deter_original_dense_df,
    #         noisy_variance=all_noisy_models,
    #         clean_variance=deter_original_pruned_df)
    #     val_dict["Feature Variance sto Val"]

# print("accuracy:{}, gradient norm: {},Hg norm {}".format(accuracy,norm_grad,norm_hg))
    if record:
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
        for n, v in val_dict.items():
            log_dict[n] = v[0]
        for n, v in test_dict.items():
            log_dict[n] = v[0]
        log_dict.update(val_dict)
        log_dict.update(test_dict)
        wandb.log(log_dict)
    t_end = time.time()
    print("Measure total time: {} s".format(t_end - t_begining))
    if record:
        return accuracy, val_dict["val_accuracy"][0]
    else:
        return val_dict, test_dict


def measure_gradient_flow_only(model: nn.Module, dataLoader, testLoader, cfg):
    model = copy.deepcopy(model)
    disable_bn(model)
    if not cfg.fine_tune_exclude_layers:
        disable_exclude_layers(model, cfg.exclude_layers)
    if not cfg.fine_tune_non_zero_weights:
        disable_all_except(model, cfg.exclude_layers)
    model.to(device=device)

    ###################################################
    # Calculate everything with respect to the validation set
    ###################################################

    val_dict = {}
    print("just before cal_grad")
    t0 = time.time()
    grad: typing.List[torch.Tensor] = cal_grad(model, trainloader=dataLoader)
    print("Just finished cal_grad")
    t1 = time.time()
    print("Time in cal_grad valset: {} s".format(t1 - t0))

    grad_vect = parameters_to_vector(grad)
    norm_grad = torch.norm(grad_vect)
    val_dict["val_set_gradient_magnitude"] = [float(norm_grad.cpu().detach().numpy())]

    ###################################################
    # Calculate everything with respect to the test set
    ###################################################

    test_dict = {}
    t0 = time.time()
    grad: typing.List[torch.Tensor] = cal_grad(model, trainloader=testLoader)
    # t0 = time.time()
    t1 = time.time()
    print("Time in cal_grad test_set: {} s".format(t1 - t0))

    grad_vect = parameters_to_vector(grad)
    norm_grad = torch.norm(grad_vect)
    test_dict["test_set_gradient_magnitude"] = [float(norm_grad.cpu().detach().numpy())]

    return test_dict, val_dict


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
