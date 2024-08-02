import logging
import os
import argparse
from truncated_models import *
import HRankPlus.utils.common as hrutils
import time
from pathlib import Path
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import random_split
from truncated_models.small_models import test_with_modified_network
import datetime
import omegaconf
import torch.optim as optim
import pandas as pd
from sparse_ensemble_utils import disable_bn
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: {}".format(device))


def get_logger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def train_probes_with_logger(epoch, train_loader, model, criterion, optimizers, schedulers=None, logger=None,
                             number_losses=7, is_ffcv=False):
    batch_time = hrutils.AverageMeter('Time', ':6.3f')
    data_time = hrutils.AverageMeter('Data', ':6.3f')
    losses_list = []
    top1_list = []
    top5_list = []
    for i in range(number_losses):
        losses = hrutils.AverageMeter('Loss prob {}'.format(i), ':.4e')
        top1 = hrutils.AverageMeter('Acc@1 prob {}'.format(i), ':6.2f')
        top5 = hrutils.AverageMeter('Acc@5 prob {}'.format(i), ':6.2f')
        losses_list.append(losses)
        top1_list.append(top1)
        top5_list.append(top5)

    model.train()
    disable_bn(model)
    batch = next(iter(train_loader))
    batch_size = batch[0].shape[0]
    # print_freq = ((len(train_loader) // batch_size) // 5)  # // batch_size
    print_freq = ((256 * 50) // batch_size)  # // batch_size
    del batch_size
    # print_freq = 1
    end = time.time()

    for param_group in optimizers[0].param_groups:
        cur_lr = param_group['lr']
    if logger:
        logger.info('learning_rate: ' + str(cur_lr))
    else:
        print('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    for batch_index, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if not is_ffcv:
            images = images.cuda()
            target = target.cuda()

        # compute outputy
        intermideate_logits, final_logits = model(images)
        assert len(
            intermideate_logits) == number_losses, " The number of losses passed {} is not equal to the number of intermediate logits of the model {}".format(
            number_losses, len(intermideate_logits))
        all_loss = []
        for logits in intermideate_logits:
            loss_i = criterion(logits, target)
            all_loss.append(loss_i)

        n = images.size(0)
        # measure accuracy and record loss
        for i, logits in enumerate(intermideate_logits):
            prec1, prec5 = hrutils.accuracy(logits, target, topk=(1, 5))
            losses_list[i].update(all_loss[i].item(), n)  # accumulated loss
            top1_list[i].update(prec1.item(), n)
            top5_list[i].update(prec5.item(), n)

        # compute gradient and do SGD step
        for optimizer in optimizers:
            optimizer.zero_grad()
        for i, loss in enumerate(all_loss):
            #
            # if i < len(all_loss) - 1:
            #     loss.backward()
            # else:
            loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print("Have finished batch {} out of {} with time processing {}".format(batch_index, num_iter, batch_time.avg))

        torch.cuda.empty_cache()
        gc.collect()

        if batch_index % print_freq == 0:
            if logger:
                logger.info('Epoch[{0}]({1}/{2}):'.format(epoch, batch_index, num_iter))
                logger.info("Average Time per batch {}, Average time loading:{}".format(batch_time.avg, data_time.avg))
            else:
                print('Epoch[{0}]({1}/{2}):'.format(epoch, batch_index, num_iter))
            for i in range(len(losses_list)):
                loss_i = losses_list[i]
                top1_i = top1_list[i]
                top5_i = top5_list[i]
                if logger:
                    logger.info(
                        "(Prob index {index}) Loss  {loss.avg:.4f} Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}".format(
                            index=i,
                            loss=loss_i, top1=top1_i, top5=top5_i))
                else:
                    print("(Prob index {index}) Loss  {loss.avg:.4f} Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}".format(
                        index=i,
                        loss=loss_i, top1=top1_i, top5=top5_i))

    if schedulers:
        for scheduler in schedulers:
            scheduler.step()

    return losses_list, top1_list, top5_list


def validate_with_logger(epoch, val_loader, model, criterion, args, logger=None, number_losses=7):
    batch_time = hrutils.AverageMeter('Time', ':6.3f')
    # losses = hrutils.AverageMeter('Loss', ':.4e')
    # top1 = hrutils.AverageMeter('Acc@1', ':6.2f')
    # top5 = hrutils.AverageMeter('Acc@5', ':6.2f')
    losses_list = []
    top1_list = []
    top5_list = []
    final_loss = hrutils.AverageMeter('Loss final output', ':.4e')
    final_top1 = hrutils.AverageMeter('Acc@1 final output ', ':6.2f')
    final_top5 = hrutils.AverageMeter('Acc@5 final output ', ':6.2f')
    for i in range(number_losses):
        losses = hrutils.AverageMeter('Loss prob {}'.format(i), ':.4e')
        top1 = hrutils.AverageMeter('Acc@1 prob {}'.format(i), ':6.2f')
        top5 = hrutils.AverageMeter('Acc@5 prob {}'.format(i), ':6.2f')
        losses_list.append(losses)
        top1_list.append(top1)
        top5_list.append(top5)

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute outputs
            all_logits, final_logits = model(images)
            n = images.size(0)

            final_loss.update(criterion(final_logits, target).item(), n)
            final_prec1, final_prec5 = hrutils.accuracy(final_logits, target, topk=(1, 5))

            final_top1.update(final_prec1.item(), n)
            final_top5.update(final_prec5.item(), n)

            all_loss = []
            for logits in all_logits:
                loss_i = criterion(logits, target)
                all_loss.append(loss_i)

            # measure accuracy and record loss
            for i, logits in enumerate(all_logits):
                prec1, prec5 = hrutils.accuracy(logits, target, topk=(1, 5))
                losses_list[i].update(all_loss[i].item(), n)  # accumulated loss
                top1_list[i].update(prec1.item(), n)
                top5_list[i].update(prec5.item(), n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    for i in range(len(losses_list)):
        loss_i = losses_list[i]
        top1_i = top1_list[i]
        top5_i = top5_list[i]
        if logger:
            logger.info(
                "(Prob index {index}) Loss  {loss.avg:.4f} Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}".format(
                    index=i,
                    loss=loss_i, top1=top1_i, top5=top5_i))
        else:
            print("(Prob index {index}) Loss  {loss.avg:.4f} Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}".format(
                index=i,
                loss=loss_i, top1=top1_i, top5=top5_i))
    if logger:
        logger.info(
            "(Final output) Loss  {loss.avg:.4f} Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}".format(
                loss=final_loss, top1=final_top1, top5=final_top5))
    else:
        print("(Final output) Loss  {loss.avg:.4f} Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}".format(
            loss=final_loss, top1=final_top1, top5=final_top5))
    # logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    torch.cuda.empty_cache()
    gc.collect()
    #             .format(top1=top1, top5=top5))

    return losses_list, top1_list, top5_list, final_loss, final_top1, final_top5


def main(args):
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    print("Normal data loaders loaded!!!!")

    cifar10_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    cifar100_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    stats_to_use = cifar10_stats if args.dataset == "cifar10" else cifar100_stats
    # Data
    print('==> Preparing data..')
    current_directory = Path().cwd()
    data_path = "."
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        data_path = "/nobackup/sclaam/data"
    elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
        data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
    elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
        data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
    elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
        data_path = "/home/luisaam/Documents/PhD/data/"
    print(data_path)
    batch_size = args.batch_size
    if "32" in args.name:
        batch_size = 32
    if "64" in args.name:
        batch_size = 64

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

        cifar10_train, cifar10_val = random_split(trainset, [len(trainset) - args.eval_size, args.eval_size])

        trainloader = torch.utils.data.DataLoader(
            cifar10_train, batch_size=128, shuffle=True, num_workers=args.num_workers)

        valloader = torch.utils.data.DataLoader(
            cifar10_val, batch_size=128, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
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
        if args.modeltype1 == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, RF_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, RF_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "small_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet50":
        if args.modeltype1 == "normal" and args.dataset == "cifar10":
            net = truncated_ResNet50_rf(num_classes=10, rf_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "cifar100":
            net = truncated_ResNet50_rf(num_classes=100, rf_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "tiny_imagenet":
            net = truncated_ResNet50_rf(num_classes=200, rf_level=args.RF_level)

        if args.modeltype1 == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.modeltype1 == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg19":
        if args.modeltype1 == "normal" and args.dataset == "cifar10":
            net = truncated_VGG_RF("VGG19_rf", num_classes=10, RF_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "cifar100":
            net = truncated_VGG_RF("VGG19_rf", num_classes=100, RF_level=args.RF_level)

        if args.modeltype1 == "normal" and args.dataset == "tiny_imagenet":
            net = truncated_VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "small_imagenet":
            net = truncated_VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet24":
        if args.modeltype1 == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=args.RF_level)
        if args.modeltype1 == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=args.RF_level)
        if args.modeltype1 == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.modeltype1,
                                                                                     args.dataset))
    if args.model == "resnet_small":
        if args.modeltype1 == "normal" and args.dataset == "cifar10":
            net = small_truncated_ResNet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)
        if args.modeltype1 == "normal" and args.dataset == "cifar100":
            net = small_truncated_ResNet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width)
        if args.modeltype1 == "normal" and args.dataset == "tiny_imagenet":
            net = small_truncated_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.modeltype1 == "normal" and args.dataset == "small_imagenet":
            net = small_truncated_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.modeltype1 == "pytorch" and args.dataset == "cifar10":
            # raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.modeltype1 == "pytorch" and args.dataset == "cifar100":
            # raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)

    ###########################################################################
    if args.solution:

        temp_dict = torch.load(args.solution, map_location=torch.device('cpu'))["net"]
        if args.modeltype1 == "normal" and args.RF_level != 0:
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

    net = net.to(device)
    x, y = next(iter(testloader))
    x = x.to(device)
    net.eval()
    intermediate_pred, final_pred = net(x[0].view(1, 3, x[0].size(-1), x[0].size(-1)))
    num_losses = len(intermediate_pred)
    modules_true, modules_false = net.set_fc_only_trainable()

    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizers = []
    # schedulers = []
    for module in modules_true:
        optimizer = optim.Adam(module.parameters(), lr=args.lr, weight_decay=5e-3)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        optimizers.append(optimizer)
        # schedulers.append(scheduler)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    additional_path_name = "{model}/{dataset}/{RF_level}".format(model=args.model, dataset=args.dataset,
                                                                 RF_level=args.RF_level)

    # additional_path_name = "{model}/{dataset}/{RF_level}/{now}_{seed}".format(model=args.model, dataset=args.dataset,
    #                                                                           RF_level=args.RF_level, now=now,
    #                                                                           seed=args.seedname1)

    run_file_path = "{}/{}".format(args.job_dir, additional_path_name)
    Path(run_file_path).mkdir(parents=True, exist_ok=True)
    logger = hrutils.get_logger("{}/{}/logger_{}.log".format(args.job_dir, additional_path_name, now))
    filepath = "{}/{}/{}_{}_logits_probs.csv".format(args.job_dir, additional_path_name, args.seedname1, now)
    name = "{}_{}".format(args.seedname1, now)
    args.job_dir = run_file_path
    hrutils.record_config(args)

    for epoch in range(args.epochs):

        loss_list_epoch, top1_list_epoch, top5_list_epoch = train_probes_with_logger(epoch, trainloader, net, criterion,
                                                                                     optimizers,
                                                                                     logger=logger,
                                                                                     number_losses=num_losses,
                                                                                     is_ffcv=args.ffcv)
        loss_list_epoch_val, top1_list_epoch_val, top5_list_epoch_val, final_loss, final_top1, final_top5 = validate_with_logger(
            epoch, testloader, net,
            criterion, args,
            logger=logger,
            number_losses=num_losses)

        if Path(filepath).is_file():

            log_dict = {"Epoch": [epoch]}
            for i, top1_acc in enumerate(top1_list_epoch):
                log_dict["Prob {} train accuracy".format(i)] = top1_acc.avg
            for i, top1_acc in enumerate(top1_list_epoch_val):
                log_dict["Prob {} test accuracy".format(i)] = top1_acc.avg
            log_dict["Final test accuracy"] = final_top1

            df = pd.DataFrame(log_dict)
            df.to_csv(filepath, mode="a", header=False, index=False)
        else:
            # Try to read the file to see if it is
            log_dict = {"Epoch": [epoch]}
            for i, top1_acc in enumerate(top1_list_epoch):
                log_dict["Prob {} train accuracy".format(i)] = top1_acc.avg
            for i, top1_acc in enumerate(top1_list_epoch_val):
                log_dict["Prob {} test accuracy".format(i)] = top1_acc.avg
            log_dict["Final test accuracy"] = final_top1
            df = pd.DataFrame(log_dict)
            df.to_csv(filepath, sep=",", index=False)

        # hrutils.save_checkpoint({
        #     'epoch': epoch,
        #     'net': compress_model.state_dict(),
        #     'best_top1_acc': best_top1_acc,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best, args.job_dir)

        state = {
            'net': net.state_dict(),
            'acc': top1_list_epoch_val[-1],
            'epoch': epoch,
            'config': args,
        }
        if not os.path.isdir(args.job_dir):
            os.mkdir(args.job_dir)

        torch.save(state, '{}/{}_logit_probes.pth'.format(args.job_dir, name))
        # if os.path.isfile('{}/{}_logit_probes.pth'.format(save_folder, name, best_acc)):
        #     os.remove('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc))


def run_local_test():
    cfg = omegaconf.DictConfig({
        "solution": "/home/luisaam/checkpoints/resnet_small_normal_small_imagenet_seed.8_rf_level_5_recording_200_test_acc_62.13.pth",
        "modeltype1": "normal",
        "seedname1": "_seed_8",
        "RF_level": 5,
        "epochs": 1,
        "ffcv": 0,
        "ffcv_val": "",
        "ffcv_train": "",
        "batch_size": 64,
        "model": "resnet_small",
        "dataset": "small_imagenet",
        "num_workers": 0,
        "input_resolution": 224,
        "width": 1,
        "name": "no_name",
        "job_dir": "truncated_results_local",
        "lr": 0.1,
        "resume": False,

    })
    main(cfg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Truncated experiments')
    # parser.add_argument('-arch', '--architecture', type=str, default="resnet18", help='Architecture for analysis',
    #                     required=True)

    parser.add_argument('-s', '--solution', type=str, default="", help='',
                        required=False)

    parser.add_argument('-sn1', '--seedname1', type=str, default="", help='',
                        required=True)
    parser.add_argument('-sn2', '--seedname2', type=str, default="", help='',
                        required=False)
    parser.add_argument('-li', '--layer_index', type=int, default=0, help='',
                        required=False)
    parser.add_argument('-e', '--experiment', type=int, default=1, help='',
                        required=False)
    parser.add_argument('-mt1', '--modeltype1', type=str, default="alternative", help='',
                        required=False)
    parser.add_argument('-mt2', '--modeltype2', type=str, default="alternative", help='',
                        required=False)
    parser.add_argument('-ft1', '--filetype1', type=str, default="npy", help='',
                        required=False)
    parser.add_argument('-ft2', '--filetype2', type=str, default="npy", help='',
                        required=False)
    parser.add_argument('-t', '--train', type=int, default=1, help='',
                        required=False)

    parser.add_argument('--RF_level', default=3, type=int, help='Receptive field of model 1')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate for the fine-tuning')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to train')
    # parser.add_argument('--RF_level2', default=3, type=int, help='Receptive field of model 2')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--dataset', default="cifar10", type=str, help='Dataset to use [cifar10,cifar100]')
    parser.add_argument('--model', default="resnet50", type=str, help='Architecture of model [resnet18,resnet50]')

    parser.add_argument('--name', '-n', default="no_name", help='name of the loss files, usually the seed name')
    # parser.add_argument('--eval_set', '-es', default="val", help='On which set to performa the calculations')
    parser.add_argument('--job_dir', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where the output of the algorithm is going to be saved')
    # parser.add_argument('--data_folder', default="/nobackup/sclaam/data", type=str,
    #                     help='Location of the dataset', required=True)
    parser.add_argument('--width', default=1, type=int, help='Width of the model')
    # parser.add_argument('--eval_size', default=1000, type=int, help='How many images to use in the calculation')
    # parser.add_argument('--batch_size', default=128, type=int, help='Batch Size for loading data')
    parser.add_argument('--input_resolution', default=224, type=int,
                        help='Input Resolution for Small ImageNet')
    parser.add_argument('--ffcv', action='store_true', help='Use FFCV loaders')

    parser.add_argument('--ffcv_train',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv",
                        type=str, help='FFCV train dataset')

    parser.add_argument('--ffcv_val',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv",
                        type=str, help='FFCV val dataset')
    parser.add_argument('--device', default="cpu", type=str, help='Which device to perform the matrix multiplication')
    parser.add_argument('--resume', action="store_true", help='Resuming training')
    # parser.add_argument('--num_losses', default=7, type=int, help='How many intermediate predictions there are')
    # parser.add_argument('--subtract_mean', default=1, type=int, help='Subtract mean of representations')

    args = parser.parse_args()

    if args.experiment == 1:
        main(args)

    # run_local_test()
