import copy
import os
import time
import wandb
import omegaconf
import pandas as pd
import pickle
from thop import profile
import argparse
import torch
import torch.nn as nn
import optuna
from delve import SaturationTracker

from main import prune_function, remove_reparametrization, get_layer_dict, get_datasets, count_parameters
from alternate_models import *
from sparse_ensemble_utils import test
from pathlib import Path
from sam import SAM
from shrinkbench.metrics.flops import flops
from KFAC_Pytorch.optimizers import KFACOptimizer, EKFACOptimizer
from train_CIFAR10 import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:{}".format(device))


# def measure_flops_cost(net, trainloader, testloader, optimizer, file_name_sufix, surname="", epochs=40,
#                        record_time=False,
#                        save_folder="", use_scheduler=False, use_scheduler_batch=False, save=False, record=False,
#                        verbose=0,
#                        grad_clip=0, macs_per_batch=None):
#     net.to(device)
#
#     if macs_per_batch:
#         total_training_macs = 0
#
#     for epoch in range(epochs):  # loop over the dataset multiple times
#
#         running_loss = 0.0
#
#         correct = 0
#         total = 0
#
#         # for i, data in enumerate(trainloader, 0):
#         for i in range(len(trainloader)):
#             # get the inputs; data is a list of [inputs, labels]
#             # zero the parameter gradients
#             if isinstance(optimizer, KFACOptimizer) or isinstance(optimizer, EKFACOptimizer):
#                 if macs_per_batch:
#                     forward_pass = macs_per_batch
#                     backward_pass = 8 * macs_per_batch
#                     total_training_macs = forward_pass + backward_pass
#
#             if isinstance(optimizer, ASAM):
#
#                 if macs_per_batch:
#                     forward_pass = macs_per_batch
#                     backward_pass = 2 * macs_per_batch
#                     total_training_macs = 2 * (forward_pass + backward_pass)
#
#         if record:
#             filepath = "{}/{}.csv".format(save_folder, file_name_sufix)
#             if Path(filepath).is_file():
#                 log_dict = {"Epoch": [epoch], "test accuracy": [test_accuracy], "training accuracy": [train_accuracy]}
#                 df = pd.DataFrame(log_dict)
#                 df.to_csv(filepath, mode="a", header=False, index=False)
#             else:
#                 # Try to read the file to see if it is
#                 log_dict = {"Epoch": [epoch], "test accuracy": [test_accuracy], "training accuracy": [train_accuracy]}
#                 df = pd.DataFrame(log_dict)
#                 df.to_csv(filepath, sep=",", index=False)
#
#     return best_acc


def training(net, trainloader, testloader, optimizer, file_name_sufix, config, surname="", epochs=40, record_time=False,
             save_folder="", use_scheduler=False, use_scheduler_batch=False, save=False, record=False, verbose=0,
             grad_clip=0, saturationTracker=False, record_flops=False, macs_per_batch=None, flops_per_batch=None):
    criterion = nn.CrossEntropyLoss()
    args = config
    net.to(device)

    if saturationTracker:
        csv_tracker = SaturationTracker("{}/{}".format(save_folder, file_name_sufix), save_to="csv", modules=net,
                                        device=device)
        # plot_tracker = SaturationTracker("{}/{}".format(save_folder, file_name_sufix), save_to="plot", modules=net,
        #                                  device=device)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0

    if record_flops:
        assert macs_per_batch is not None, "You cannot set record_flops=True and not specify macs_per_batch"
        assert flops_per_batch is not None, "You cannot set record_flops=True and not specify flops_per_batch"
        total_training_macs = 0
        total_training_flops = 0
        forward_call_macs = macs_per_batch
        forward_call_flops = flops_per_batch
        backward_call_macs = macs_per_batch * 2
        backward_call_flops = flops_per_batch * 2
    for epoch in range(epochs):  # loop over the dataset multiple times

        t0 = time.time_ns()
        running_loss = 0.0

        net.train()
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            if isinstance(optimizer, KFACOptimizer) or isinstance(optimizer, EKFACOptimizer):
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                if optimizer.steps % optimizer.TCov == 0:
                    # compute true fisher
                    optimizer.acc_stats = True
                    with torch.no_grad():
                        sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                                      1).squeeze().to(device)
                    loss_sample = criterion(outputs, sampled_y)
                    loss_sample.backward(retain_graph=True)
                    optimizer.acc_stats = False
                    optimizer.zero_grad()  # clear the gradient for computing true-fisher.

                loss.backward()
                # if macs_per_batch:
                #     forward_pass = macs_per_batch
                #     backward_pass = 4*macs_per_batch
                #     total_training_macs = forward_pass+backward_pass

                if grad_clip:
                    nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.step()

                if record_flops:
                    total_training_macs += (forward_call_macs + 4 * backward_call_macs)
                    total_training_flops += (forward_call_flops + 4 * backward_call_flops)
                #
                # if record_function_calls:
                #     with open(file_name_sufix + "/function_call_" + surname + ".txt", "a") as f:
                #         f.write("2\n")

                item = loss.item()
                running_loss += item

                if torch.all(outputs > 0):
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    soft_max_outputs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(soft_max_outputs, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()

                if verbose == 1:
                    print("Running loss: {}".format(running_loss))
                # with open(file_name_sufix + f"/loss_training_{surname}.txt", "a") as f:
                #     f.write(f"{item}\n")
                if use_scheduler and use_scheduler_batch:
                    scheduler.step()

            if isinstance(optimizer, SAM):

                # first forward-backward pass
                # print("batch:{}".format(i))
                # print(net(inputs).shape)
                # print(labels.shape)
                outputs = net(inputs)
                loss = criterion(outputs, labels)  # use this loss for any training statistics
                loss.backward()
                optimizer.first_step(zero_grad=True)
                # print(loss.item())
                # second forward-backward pass
                criterion(net(inputs), labels).backward()  # make sure to do a full forward pass
                if grad_clip:
                    nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.second_step(zero_grad=True)

                if torch.all(outputs > 0):
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    soft_max_outputs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(soft_max_outputs, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()

                if record_flops:
                    total_training_macs += 2 * (forward_call_macs + backward_call_macs)
                    total_training_flops += 2 * (forward_call_flops + backward_call_flops)

                # def closure():
                #     loss = criterion(labels, net(inputs))
                #     loss.backward()
                #     return loss

                # loss = criterion(labels, net(inputs))

                # loss.backward()

                # optimizer.step()

        t1 = time.time_ns()
        use_cuda = True if torch.cuda.is_available() else False
        # train_accuracy = test(net, use_cuda=use_cuda, testloader=trainloader, verbose=0)
        train_accuracy = 100. * correct / total
        test_accuracy = test(net, use_cuda=use_cuda, testloader=testloader, verbose=0)
        if verbose == 2:
            print("Test Accuracy at Epoch {}:{}".format(epoch, test_accuracy))

        if use_scheduler and not use_scheduler_batch:
            scheduler.step()

        if test_accuracy > best_acc:

            if save:

                state = {
                    'net': net.state_dict(),
                    'acc': test_accuracy,
                    'epoch': epoch,
                    "cfg": omegaconf.OmegaConf.to_container(omegaconf.DictConfig(vars(args)), resolve=True),

                }

                if not os.path.isdir(save_folder):
                    os.mkdir(save_folder)

                if os.path.isfile('{}/{}_test_acc_{}.pth'.format(save_folder, file_name_sufix, best_acc)):
                    os.remove('{}/{}_test_acc_{}.pth'.format(save_folder, file_name_sufix, best_acc))

                torch.save(state, '{}/{}_test_acc_{}.pth'.format(save_folder, file_name_sufix, test_accuracy))

            print("Best Test Accuracy at Epoch {}:{}".format(epoch, test_accuracy))
            best_acc = test_accuracy

        if record_time:
            filepath = "{}/{}_training_time.csv".format(save_folder, file_name_sufix)
            if Path(filepath).is_file():
                log_dict = {"Epoch": [epoch], "training_time": [t1 - t0]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                # Try to read the file to see if it is
                log_dict = {"Epoch": [epoch], "training_time": [t1 - t0]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, sep=",", index=False)
        if record:
            filepath = "{}/{}_test_acc.csv".format(save_folder, file_name_sufix)
            if Path(filepath).is_file():
                log_dict = {"Epoch": [epoch], "test accuracy": [test_accuracy], "training accuracy": [train_accuracy]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                # Try to read the file to see if it is
                log_dict = {"Epoch": [epoch], "test accuracy": [test_accuracy], "training accuracy": [train_accuracy]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, sep=",", index=False)
        if record_flops:
            filepath = "{}/{}_flops.csv".format(save_folder, file_name_sufix)
            if Path(filepath).is_file():
                log_dict = {"Epoch": [epoch], "flops": [total_training_flops], "mac": [total_training_macs]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                # Try to read the file to see if it is
                log_dict = {"Epoch": [epoch], "flops": [total_training_flops], "mac": [total_training_macs]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, sep=",", index=False)

        if saturationTracker:
            # add some additional metrics we want to keep track of
            csv_tracker.add_scalar("test_accuracy", test_accuracy)
            csv_tracker.add_scalar("Epoch", epoch)
            # csv_tracker.add_scalar("loss", test_loss / total)

            # plot_tracker.add_scalar("test_accuracy", test_accuracy)
            # plot_tracker.add_scalar("loss", test_loss / total)

            # add saturations to the mix
            csv_tracker.add_saturations()
            # plot_tracker.add_saturations()

            # close the tracker to finish training
    if saturationTracker:
        csv_tracker.close()
        # plot_tracker.close()

    return best_acc



def main(args):

    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": args.model,
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": args.dataset,
         "batch_size": args.batch_size,
         "num_workers": args.num_workers,
         "noise": "gaussian",
         "input_resolution":args.input_resolution,
         "pad":args.pad,
         })

    trainloader, valloader, testloader = get_datasets(cfg)

    net = get_model(args)

    # from torchvision.models import resnet18, resnet50
    #
    # if args.model == "resnet18":
    #     if args.type == "normal" and args.dataset == "cifar10":
    #         net = ResNet18_rf(num_classes=10, RF_level=args.RF_level)
    #     if args.type == "normal" and args.dataset == "cifar100":
    #         net = ResNet18_rf(num_classes=100, RF_level=args.RF_level)
    #     if args.type == "normal" and args.dataset == "tiny_imagenet":
    #         net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
    # if args.model == "resnet50":
    #     if args.type == "normal" and args.dataset == "cifar10":
    #         net = ResNet50_rf(num_classes=10, rf_level=args.RF_level)
    #     if args.type == "normal" and args.dataset == "cifar100":
    #         net = ResNet50_rf(num_classes=100, rf_level=args.RF_level)
    #     if args.type == "normal" and args.dataset == "tiny_imagenet":
    #         net = ResNet50_rf(num_classes=200, rf_level=args.RF_level)
    #     if args.type == "pytorch" and args.dataset == "cifar10":
    #         net = resnet50()
    #         in_features = net.fc.in_features
    #         net.fc = nn.Linear(in_features, 10)
    #     if args.type == "pytorch" and args.dataset == "cifar100":
    #         net = resnet50()
    #         in_features = net.fc.in_features
    #         net.fc = nn.Linear(in_features, 100)
    # if args.model == "vgg19":
    #     if args.type == "normal" and args.dataset == "cifar10":
    #         net = VGG_RF("VGG19_rf", num_classes=10, RF_level=args.RF_level)
    #     if args.type == "normal" and args.dataset == "cifar100":
    #         net = VGG_RF("VGG19_rf", num_classes=100, RF_level=args.RF_level)
    #     if args.type == "normal" and args.dataset == "tiny_imagenet":
    #         net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
    #     if args.type == "normal" and args.dataset == "small_imagenet":
    #         net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
    # if args.model == "resnet24":
    #
    #     if args.type == "normal" and args.dataset == "cifar10":
    #         net = ResNet24_rf(num_classes=10, rf_level=args.RF_level)
    #     if args.type == "normal" and args.dataset == "cifar100":
    #         net = ResNet24_rf(num_classes=100, rf_level=args.RF_level)
    #     if args.type == "normal" and args.dataset == "tiny_imagenet":
    #         net = ResNet24_rf(num_classes=200, rf_level=args.RF_level)
    #     if args.type == "pytorch" and args.dataset == "cifar10":
    #         # # net = resnet50()
    #         # # in_features = net.fc.in_features
    #         # net.fc = nn.Linear(in_features, 10)
    #         raise NotImplementedError(
    #             " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
    #                                                                                  args.dataset))
    # if args.model == "vgg_small_deep":
    #
    #     if args.type == "normal" and args.dataset == "cifar10":
    #         net = DeepSmallVGG_RF("small_vgg", num_classes=10, RF_level=args.RF_level)
    #
    #     if args.type == "normal" and args.dataset == "cifar100":
    #         net = DeepSmallVGG_RF("small_vgg", num_classes=100, RF_level=args.RF_level)
    #
    #     if args.type == "normal" and args.dataset == "tiny_imagenet":
    #         net = DeepSmallVGG_RF("small_vgg", num_classes=200, RF_level=args.RF_level)
    #
    #     if args.type == "normal" and args.dataset == "small_imagenet":
    #         net = DeepSmallVGG_RF("deep_small_vgg", num_classes=200, RF_level=args.RF_level)
    # if args.model == "resnet_small":
    #     if args.type == "normal" and args.dataset == "cifar10":
    #         net = small_ResNet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)
    #
    #     if args.type == "normal" and args.dataset == "cifar100":
    #         net = small_ResNet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width)
    #     if args.type == "normal" and args.dataset == "tiny_imagenet":
    #         net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
    #     if args.type == "normal" and args.dataset == "small_imagenet":
    #         net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
    #     if args.type == "pytorch" and args.dataset == "cifar10":
    #         raise NotImplementedError
    #         # net = resnet50()
    #         # in_features = net.fc.in_features
    #         # net.fc = nn.Linear(in_features, 10)
    #     if args.type == "pytorch" and args.dataset == "cifar100":
    #         raise NotImplementedError
    # if args.model == "resnet40_small":
    #     if args.type == "normal" and args.dataset == "cifar10":
    #         net = deep_small_ResNet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)
    #     if args.type == "normal" and args.dataset == "cifar100":
    #         net = deep_small_ResNet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width)
    #     if args.type == "normal" and args.dataset == "tiny_imagenet":
    #         net = deep_small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
    #     if args.type == "normal" and args.dataset == "small_imagenet":
    #         net = deep_small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
    #     if args.type == "pytorch" and args.dataset == "cifar10":
    #         raise NotImplementedError
    #         # net = resnet50()
    #         # in_features = net.fc.in_features
    #         # net.fc = nn.Linear(in_features, 10)
    #     if args.type == "pytorch" and args.dataset == "cifar100":
    #         # net = resnet50()
    #         # in_features = net.fc.in_features
    #         # net.fc = nn.Linear(in_features, 100)
    #         raise NotImplementedError
    #         # net = resnet50()
    #         # in_features = net.fc.in_features
    #         # net.fc = nn.Linear(in_features, 100)
    # if args.model == "resnet25_small":
    #     if args.type == "normal" and args.dataset == "cifar10":
    #         net = deep_2_small_Resnet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width,
    #                                      number_layers=25)
    #     if args.type == "normal" and args.dataset == "cifar100":
    #         net = deep_2_small_Resnet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width,
    #                                      number_layers=25)
    #     if args.type == "normal" and args.dataset == "tiny_imagenet":
    #         net = deep_2_small_Resnet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width,
    #                                      number_layers=25)
    #     if args.type == "normal" and args.dataset == "small_imagenet":
    #         net = deep_2_small_Resnet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width,
    #                                      number_layers=25)
    #     if args.type == "pytorch" and args.dataset == "cifar10":
    #         raise NotImplementedError
    #         # net = resnet50()
    #         # in_features = net.fc.in_features
    #         # net.fc = nn.Linear(in_features, 10)
    #     if args.type == "pytorch" and args.dataset == "cifar100":
    #         # net = resnet50()
    #         # in_features = net.fc.in_features
    #         # net.fc = nn.Linear(in_features, 100)
    #         raise NotImplementedError
    #         # net = resnet50()
    #         # in_features = net.fc.in_features
    #         # net.fc = nn.Linear(in_features, 100)
    #
    #
    #     ########################################################################
    #     #                            Optimiser
    #     ########################################################################

    seed = time.time()

    if args.optimiser == "kfac":
        # optimiser = KFACOptimizer(net, lr=args.lr, momentum=args.momentum, weight_decay=0.003, damping=0.03)
        optimiser = KFACOptimizer(net, lr=args.lr, momentum=args.momentum, weight_decay=5e-4, damping=0.03)
    if args.optimiser == "ekfac":
        # optimiser = EKFACOptimizer(net, lr=args.lr, momentum=args.momentum, weight_decay=0.003, damping=0.03)
        optimiser = EKFACOptimizer(net, lr=args.lr, momentum=args.momentum, weight_decay=5e-4, damping=0.03)
    if args.optimiser == "sam":
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimiser = SAM(net.parameters(), base_optimizer, lr=args.lr, momentum=args.momentum, rho=0.5, adaptive=True,
                        weight_decay=5e-4)
    solution_name = "{}_{}_{}_rf_level_{}_{}_{}".format(args.model, args.type, args.dataset, args.RF_level,
                                                     args.name,seed)
    if args.save:
        state = {
            'net': net.state_dict(),
            'acc': 0,
            'epoch': -1,
            "cfg": omegaconf.OmegaConf.to_container(omegaconf.DictConfig(vars(args)), resolve=True),
        }
        torch.save(state, '{}/{}_initial_weights.pth'.format(args.save_folder, solution_name))

    t0 = time.time()

    total_flops = 0
    x = None
    y = None
    net.to(device)
    macs_batch = None
    batch_flops = None

    if args.record_flops:
        x, y = next(iter(trainloader))
        x = x.to(device)
        batch_flops, _ = flops(net, x)
        input = torch.randn(1, 3,args.input_resolution ,args.input_resolution)
        input = input.to(device)
        macs_one_image, params = profile(copy.deepcopy(net), inputs=(input,))
        macs_batch = macs_one_image * args.batch_size

    best_accuracy = training(net, trainloader, testloader, optimiser, solution_name, epochs=args.epochs,
                             save_folder=args.save_folder, use_scheduler=args.use_scheduler, save=args.save,
                             record=args.record, verbose=2, grad_clip=args.grad_clip, record_time=args.record_time,
                             record_flops=args.record_flops,
                             macs_per_batch=macs_batch, flops_per_batch=batch_flops,
                             saturationTracker=args.record_saturation, config=args)
    t1 = time.time()
    training_time = t1 - t0
    print("Training time: {}".format(training_time))
    return best_accuracy, training_time


def optuna_optimization(args):
    wandb.init(
        entity="luis_alfredo",
        config=omegaconf.OmegaConf.to_container(omegaconf.DictConfig(vars(args)), resolve=True),
        project="Receptive_Field",
        name="{} parameter optimisation".format(args.optimiser),
        save_code=False,
    )

    def objective(trial):
        lr = trial.suggest_float('lr', 0.00001, 0.5)
        momentum = trial.suggest_float('momentum', 0.3, 1)
        gradient_clip = trial.suggest_float('grad_clip', -0.1, 1.1)

        # optimiser_type = trial.suggest_categorical("optimiser", ["kfac", "ekfac"])
        # use_scheduler = trial.suggest_categorical("use_scheduler", [False, True])
        cfg = omegaconf.DictConfig({
            "dataset": "cifar10",
            "RF_level": 2,
            "lr": lr,
            "momentum": momentum,
            "model": "resnet50",
            "optimiser": args.optimiser,
            "epochs": 5,
            "use_scheduler": False,
            "save": False,
            "save_folder": args.save_folder,
            "num_workers": 0,
            "type": "normal",
            "folder": "",
            "record": False,
            "name": "hyper_optim",
            "batch_size": 32,
            "use_scheduler_batch": False,
            "grad_clip": gradient_clip,

        })
        acc = -1
        try:

            acc, train_time = main(cfg)

            wandb.log({"acc": acc, "train_time": train_time, "momentum": momentum, "grad_clip": gradient_clip,
                       "initial_lr": lr})


        except Exception as e:

            print(e)
            acc = 0

        return acc

    search_space = {"lr": [0.1, 0.01, 0.001], "momentum": [0.99, 0.9, 0.7, 0.5],
                    "grad_clip": [1, 0.8, 0.7, 0.5, 0.1, 0]}

    if os.path.isfile("second_order_{}_hyperparameter_optimization.pkl".format(args.optimiser)):

        with open("second_order_{}_hyperparameter_optimization.pkl".format(args.optimiser), "rb") as f:
            study = pickle.load(f)
    else:
        study = optuna.create_study(direction="maximize",
                                    study_name="second_order_{}_hyperparameter_optimization".format(args.optimiser),
                                    sampler=optuna.samplers.GridSampler(search_space))

    study.optimize(objective, n_trials=4 * 4 * 3, gc_after_trial=True, n_jobs=1)

    trials = study.best_trials

    wandb.finish()
    with open("second_order_{}_hyperparameter_optimization.pkl".format(args.optimiser), "wb") as f:

        pickle.dump(study, f)
    print("Size of the pareto front: {}".format(len(trials)))

    for trial in trials:
        f1 = trial.value
        lr, momentum, grad_clip = trial.params["lr"], trial.params["momentum"], trial.params["grad_clip"]
        print("  Value: {}".format(f1))
        print_param = omegaconf.DictConfig({
            "lr": lr,
            "momentum": momentum,
            "grad_clip": grad_clip,
            "optimiser": args.optimiser,
        })
        print("Parameters:\n\t")
        print(omegaconf.OmegaConf.to_yaml(print_param))


def run_local_test(level):
    cfg = omegaconf.DictConfig({
        "model": "resnet50",
        "dataset": "cifar10",
        "type": "normal",
        "RF_level": level,
        "lr": 0.1,
        "grad_clip": 1,
        "momentum": 0.9,
        "num_workers": 0,
        "optimiser": "sam",
        "record": False,
        "record_flops": True,
        "record_time": False,
        "use_scheduler_batch": False,
        "use_scheduler": True,
        "batch_size": 128,
        "epochs": 1,
        "name": "no_name",
        "save": False,
        "save_folder": "./second_order_results",
        "record_saturation": True

    })
    main(cfg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Second Order and Receptive field experiments')
    parser.add_argument('--experiment', default=1, type=int, help='Experiment to perform')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
    parser.add_argument('--grad_clip', default=0.1, type=float, help='Gradient clipping')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--type', default="normal", type=str, help='Type of implementation [normal,official]')
    parser.add_argument('--RF_level', default="4", type=str, help='Receptive field level')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', "-dt", default="cifar10", type=str,
                        help='Dataset to use [cifar10,tiny_imagenet,small_imagenet]')
    parser.add_argument('--model', default="resnet50", type=str, help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--save_folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where saved models are')
    parser.add_argument('--name', default="", type=str, help='Name of the file', required=False)
    parser.add_argument('--solution', default="", type=str, help='Solution to use')
    parser.add_argument('--pruning_rate', default=0.9, type=float, help='Pruning rate')
    parser.add_argument('--pad', default=0, type=int,
                        help='Pad the image to the input size ')
    parser.add_argument('--input_resolution', default=224, type=int,
                        help='Input Resolution for the dataset')
    parser.add_argument('--width', default=1, type=int, help='Width of the Network')
    parser.add_argument('--epochs', default=50, type=int, help='Epochs to train')
    parser.add_argument('--optimiser', default="sam", type=str, help='Optimiser to use')
    parser.add_argument('--save', default=0, type=int, help="Save the best model")
    parser.add_argument('--record', default=0, type=int, help="Record the test/training accuracy")
    parser.add_argument('--record_time', default=0, type=int, help="Record the training time")
    parser.add_argument('--record_flops', '-r', type=int, help='Count the flops of training')
    parser.add_argument('--record_saturation', type=int, help='Calculate saturation')
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size for training/testing")
    parser.add_argument('--use_scheduler', default=1, type=int, help="Use sine scheduler")
    parser.add_argument('--use_scheduler_batch', default=1, type=int,
                        help="Use scheduler for batches instead of epochs")

    args = parser.parse_args()

    try:

        args.RF_level = int(args.RF_level)

    except Exception as e:

        pass
    if args.experiment == 1:
        print(args)

        main(args)

    if args.experiment == 2:
        optuna_optimization(args)

    # parser = argparse.ArgumentParser(description='Second Order and Receptive field experiments')
    # parser.add_argument('--level', default=1, type=int, help="RF level")
    # args = parser.parse_args()

    # run_local_test()
