import os
import time
import wandb
import torch
import torch.nn as nn
import optuna
from main import prune_function, remove_reparametrization, get_layer_dict, get_datasets, count_parameters
from KFAC_Pytorch.optimizers import KFACOptimizer, EKFACOptimizer
import argparse
import omegaconf
from alternate_models import *
from sparse_ensemble_utils import test
import pickle
import pandas as pd
from pathlib import Path
from sam import SAM

device = "cuda" if torch.cuda.is_available() else "cpu"


def training(net, trainloader, testloader, optimizer, file_name_sufix, surname="", epochs=40, record_time=False,
             save_folder="", use_scheduler=False, use_scheduler_batch=False, save=False, record=False, verbose=0,
             grad_clip=0):
    criterion = nn.CrossEntropyLoss()
    net.to(device)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0

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
                t0 = time.time_ns()
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

                if grad_clip:
                    nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.step()

                t1 = time.time_ns()

                if record_time:
                    with open(file_name_sufix + "/time_" + surname + ".txt", "a") as f:
                        f.write(str(t1 - t0) + "\n")
                #
                # if record_function_calls:
                #     with open(file_name_sufix + "/function_call_" + surname + ".txt", "a") as f:
                #         f.write("2\n")

                item = loss.item()
                running_loss += item

                # if torch.all(outputs > 0):
                #     _, predicted = torch.max(outputs.data, 1)
                # else:
                #     soft_max_outputs = F.softmax(outputs, dim=1)
                #     _, predicted = torch.max(soft_max_outputs, 1)
                # total += labels.size(0)
                # correct += predicted.eq(labels.data).cpu().sum()

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
                loss = criterion(net(inputs), labels)  # use this loss for any training statistics
                loss.backward()
                optimizer.first_step(zero_grad=True)
                # print(loss.item())
                # second forward-backward pass
                criterion(net(inputs), labels).backward()  # make sure to do a full forward pass
                if grad_clip:
                    nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.second_step(zero_grad=True)
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                # def closure():
                #     loss = criterion(labels, net(inputs))
                #     loss.backward()
                #     return loss
                #
                # loss = criterion(labels, net(inputs))
                #
                # loss.backward()

                # optimizer.step()

        test_accuracy = test(net, use_cuda=True, testloader=testloader, verbose=0)
        train_accuracy = test(net, use_cuda=True, testloader=trainloader, verbose=0)
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
                }

                if not os.path.isdir(save_folder):
                    os.mkdir(save_folder)

                if os.path.isfile('{}/{}_test_acc_{}.pth'.format(save_folder, file_name_sufix, best_acc)):
                    os.remove('{}/{}_test_acc_{}.pth'.format(save_folder, file_name_sufix, best_acc))

                torch.save(state, '{}/{}_test_acc_{}.pth'.format(save_folder, file_name_sufix, test_accuracy))

            print("Best Test Accuracy at Epoch {}:{}".format(epoch, test_accuracy))
            best_acc = test_accuracy

        if record:
            filepath = "{}/{}.csv".format(save_folder, file_name_sufix)
            if Path(filepath).is_file():
                log_dict = {"Epoch": [epoch], "test accuracy": [test_accuracy], "training accuracy": [train_accuracy]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                # Try to read the file to see if it is
                log_dict = {"Epoch": [epoch], "test accuracy": [test_accuracy], "training accuracy": [train_accuracy]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, sep=",", index=False)
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

         })
    trainloader, valloader, testloader = get_datasets(cfg)

    from torchvision.models import resnet18, resnet50

    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, rf_level=args.RF_level)
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
            net = VGG_RF("VGG19_rf", num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, rf_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, rf_level=args.RF_level)
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
    if args.optimiser == "kfac":
        optimiser = KFACOptimizer(net, lr=args.lr, momentum=args.momentum, weight_decay=0.003, damping=0.03)
    if args.optimiser == "ekfac":
        optimiser = EKFACOptimizer(net, lr=args.lr, momentum=args.momentum, weight_decay=0.003, damping=0.03)
    if args.optimiser == "sam":
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimiser = SAM(net.parameters(), base_optimizer, lr=args.lr, momentum=args.momentum)

    solution_name = "{}_{}_{}_rf_level_{}_{}".format(args.model, args.type, args.dataset, args.RF_level,
                                                     args.name)
    if args.save:
        state = {
            'net': net.state_dict(),
            'acc': 0,
            'epoch': -1,
        }
        torch.save(state, '{}/{}_initial_weights.pth'.format(args.save_folder, solution_name))
    t0 = time.time()
    best_accuracy = training(net, trainloader, testloader, optimiser, solution_name, epochs=args.epochs,
                             save_folder=args.save_folder, use_scheduler=args.use_scheduler, save=args.save,
                             record=args.record, verbose=2, grad_clip=args.grad_clip)
    t1 = time.time()
    training_time = t1 - t0
    print("Training time: {}".format(training_time))
    return best_accuracy, training_time


def optuna_optimization(args):

    wandb.init(
        entity="luis_alfredo",
        config=omegaconf.OmegaConf.to_container(omegaconf.DictConfig(vars(args)), resolve=True),
        project="Receptive_Field",
        name="Optuna optimisation",
        reinit=True,
        save_code=True,
    )

    def objective(trial):
        lr = trial.suggest_float('lr', 0.00001, 0.5)
        momentum = trial.suggest_float('momentum', 0.3, 0.9)
        gradient_clip = trial.suggest_float('grad_clip', 0.1, 0.9)

        # optimiser_type = trial.suggest_categorical("optimiser", ["kfac", "ekfac"])
        # use_scheduler = trial.suggest_categorical("use_scheduler", [False, True])
        cfg = omegaconf.DictConfig({
            "dataset": "cifar10",
            "RF_level": 4,
            "lr": lr,
            "momentum": momentum,
            "model": "resnet50",
            "optimiser": args.optimiser,
            "epochs": 3,
            "use_scheduler": False,
            "save": False,
            "num_workers": 0,
            "type": "normal",
            "folder": "",
            "name": "hyper_optim",
            "batch_size": 32,
            "use_scheduler_batch": False,
            "grad_clip": gradient_clip,

        })
        try:

            acc, train_time = main(cfg)

        except Exception as e:

            print(e)

            return 0, 10000

        return acc, train_time

    search_space = {"lr": [0.1, 0.01, 0.001], "momentum": [0.99, 0.9, 0.7, 0.5], "grad_clip": [1, 0.5, 0.1, 0]}

    if os.path.isfile("second_order_hyperparameter_optimization.pkl"):

        with open("second_order_hyperparameter_optimization.pkl", "rb") as f:
            study = pickle.load(f)
    else:
        study = optuna.create_study(directions=["maximize", "minimize"],
                                    study_name="second_order_hyperparameter_optimization",
                                    sampler=optuna.samplers.GridSampler(search_space))

    study.optimize(objective, n_trials=4 * 4 * 3, gc_after_trial=True, n_jobs=1)

    trials = study.best_trials

    print("Size of the pareto front: {}".format(len(trials)))

    for trial in trials:
        f1, f2 = trial.values
        lr, momentum, optimiser, use_scheduler = trial.params["lr"], trial.params["momentum"], trial.params[
            "optimiser"], trial.params["use_scheduler"]
        print("  Values: {},{}".format(f1, f2))
        print_param = omegaconf.DictConfig({
            "lr": lr,
            "momentum": momentum,
            "optimiser": optimiser,
            "use_scheduler": use_scheduler,
        })
        print("Parameters:\n\t")
        print(omegaconf.OmegaConf.to_yaml(print_param))

    with open("second_order_hyperparameter_optimization.pkl", "wb") as f:

        pickle.dump(study, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Second Order and Receptive field experiments')
    parser.add_argument('--experiment', default=1, type=int, help='Experiment to perform')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
    parser.add_argument('--grad_clip', default=0.1, type=float, help='Gradient clipping')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--type', default="normal", type=str, help='Type of implementation [normal,official]')
    parser.add_argument('--RF_level', default=4, type=int, help='Receptive field level')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', "-dt", default="cifar10", type=str,
                        help='Dataset to use [cifar10,tiny_imagenet616gg]')
    parser.add_argument('--model', default="resnet50", type=str, help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--save_folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where saved models are')
    parser.add_argument('--name', default="", type=str, help='Name of the file', required=False)
    parser.add_argument('--solution', default="", type=str, help='Solution to use')
    parser.add_argument('--pruning_rate', default=0.9, type=float, help='Pruning rate')
    parser.add_argument('--epochs', default=50, type=int, help='Epochs to train')
    parser.add_argument('--optimiser', default="sam", type=str, help='Optimiser to use')
    parser.add_argument('--save', default=0, type=int, help="Save the best model")
    parser.add_argument('--record', default=0, type=int, help="Record the test/training accuracy")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size for training/testing")
    parser.add_argument('--use_scheduler', default=1, type=int, help="Use sine scheduler")
    parser.add_argument('--use_scheduler_batch', default=1, type=int,
                        help="Use scheduler for batches instead of epochs")

    args = parser.parse_args()

    if args.experiment == 1:
        print(args)
        main(args)
    if args.experiment == 2:
        optuna_optimization()
