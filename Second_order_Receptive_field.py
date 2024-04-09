import os
import time
import torch
import torch.nn as nn
import optuna
from main import test, prune_function, remove_reparametrization, get_layer_dict, get_datasets, count_parameters
from KFAC_Pytorch.optimizers import KFACOptimizer, EKFACOptimizer
import argparse
import omegaconf
from alternate_models import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def training(net, trainloader, testloader, optimizer, file_name_sufix, surname="", epochs=40,
             regularize=False, record_time=False, save_folder="", use_scheduler=False, save=False, verbose=0):
    criterion = nn.CrossEntropyLoss()
    net.to(device)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            if isinstance(optimizer, KFACOptimizer) or isinstance(optimizer, EKFACOptimizer):
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
                if verbose:
                    print("Running loss: {}".format(running_loss))
                # with open(file_name_sufix + f"/loss_training_{surname}.txt", "a") as f:
                #     f.write(f"{item}\n")

        test_accuracy = test(net, use_cuda=True, testloader=testloader, verbose=0)
        if verbose:
            print("Test Accuracy at Epoch {}:{}".format(epoch, test_accuracy))

        if use_scheduler:
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

            best_acc = test_accuracy

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
         "batch_size": 128,
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
        optimiser = KFACOptimizer(net, lr=args.lr, momentum=args.momentum)
    if args.optimiser == "ekfac":
        optimiser = EKFACOptimizer(net, lr=args.lr, momentum=args.momentum)
    solution_name = "{}_{}_{}_rf_level_{}_{}_".format(args.model, args.type, args.dataset, args.RF_level,
                                                      args.name, args.optimiser)
    t0 = time.time()
    best_accuracy = training(net, trainloader, testloader, optimiser, solution_name, epochs=args.epochs,
                             save_folder=args.folder, use_scheduler=args.use_scheduler)
    t1 = time.time()
    training_time = t1 - t0
    return best_accuracy, training_time


def optuna_optimization():
    def objective(trial):
        lr = trial.suggest_float('lr', 0.0001, 0.1, log=True)
        momentum = trial.suggest_float('momentum', 0.5, 0.9, log=True)
        optimiser_type = trial.suggest_categorical("optimiser", ["kfac", "ekfac"])
        use_scheduler = trial.suggest_categorical("use_scheduler", [False, True])
        cfg = omegaconf.DictConfig({
            "dataset": "tiny_imagenet",
            "RF_level": 2,
            "lr": lr,
            "momentum": momentum,
            "model": "resnet50",
            "optimiser": optimiser_type,
            "epochs": 5,
            "use_scheduler": use_scheduler,
            "save": False,
            "num_workers": 0,
            "type": "normal",
            "folder": "",
            "name":"hyper_optim",

        })

        acc, train_time = main(cfg)
        return acc, train_time

    study = optuna.create_study(directions=["maximize", "minimize"],
                                study_name="second_order_hyperparameter_optimization",
                                storage="sqlite:///second_order_hyperparameter_optimization.dep",
                                load_if_exists=True)
    study.optimize(objective, n_trials=100)

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Second Order and Receptive field experiments')
    parser.add_argument('--experiment', default=2, type=int, help='Experiment to perform')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning Rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--type', default="normal", type=str, help='Type of implementation [normal,official]')
    parser.add_argument('--RF_level', default=4, type=int, help='Receptive field level')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', default="cifar10", type=str, help='Dataset to use [cifar10,tiny_imagenet616gg]')
    parser.add_argument('--model', default="resnet50", type=str, help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where saved models are')
    parser.add_argument('--name', default="", type=str, help='Name of the file', required=False)
    parser.add_argument('--solution', default="", type=str, help='Solution to use')
    parser.add_argument('--pruning_rate', default=0.9, type=float, help='Pruning rate')
    parser.add_argument('--epochs', default=50, type=int, help='Epochs to train')
    parser.add_argument('--optimiser', default="kfac", type=str, help='Optimiser to use')
    parser.add_argument('--save', default=True, type=bool, help="Save the best model")

    args = parser.parse_args()
    if args.experiment == 1:
        main(args)
    if args.experiment == 2:
        optuna_optimization()
