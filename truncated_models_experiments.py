import logging
import argparse
from truncated_models import *
import HRankPlus.utils.common as hrutils
import time


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


def train_probes_with_logger(epoch, train_loader, model, criterion, optimizer, scheduler, logger=None, number_losses=7):
    batch_time = hrutils.AverageMeter('Time', ':6.3f')
    data_time = hrutils.AverageMeter('Data', ':6.3f')
    losses_list = []
    top1_list = []
    top5_list = []
    for i in range(number_losses + 1):
        losses = hrutils.AverageMeter('Loss prob {}'.format(i), ':.4e')
        top1 = hrutils.AverageMeter('Acc@1 prob {}'.format(i), ':6.2f')
        top5 = hrutils.AverageMeter('Acc@5 prob {}'.format(i), ':6.2f')
        losses_list.append(losses)
        top1_list.append(top1)
        top5_list.append(top5)

    model.train()
    batch = next(iter(train_loader))
    batch_size = batch[0].shape[0]
    print_freq = (256 * 50) // batch_size
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    if logger:
        logger.info('learning_rate: ' + str(cur_lr))
    else:
        print('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        all_logits = model(images)
        assert len(
            all_logits) == number_losses + 1, " The number of losses passed {} is not equal to the number of outputs of the model, -1".format(
            number_losses)
        all_loss = []
        for logits in all_logits[:-1]:
            loss_i = criterion(logits, target)
            all_loss.append(loss_i)

        n = images.size(0)
        # measure accuracy and record loss
        for i, logits in enumerate(all_logits):
            prec1, prec5 = hrutils.accuracy(logits, target, topk=(1, 5))
            # TODO: Finish the losses collection
            losses_list[i].update(all_loss[i].item(), n)  # accumulated loss
            top1_list[i].update(prec1.item(), n)
            top5_list[i].update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        for i, loss in enumerate(all_loss):

            if i < len(all_loss) - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            if logger:
                logger.info('Epoch[{0}]({1}/{2}):'.format(epoch, i, num_iter))
            else:
                print('Epoch[{0}]({1}/{2}):'.format(epoch, i, num_iter))
            for i in range(len(all_loss)):
                loss_i = all_loss[i]
                top1_i = top1_list[i]
                top5_i = top5_list[i]

                logger.info(
                    "(Prob index {index}) Loss  {loss.avg:.4f} Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}".format(
                        index=i,
                        loss=loss_i, top1=top1_i, top5=top5_i))

    scheduler.step()

    return losses_list, top1_list, top5_list


def validate_with_logger(epoch, val_loader, model, criterion, args, logger):
    batch_time = hrutils.AverageMeter('Time', ':6.3f')
    losses = hrutils.AverageMeter('Loss', ':.4e')
    top1 = hrutils.AverageMeter('Acc@1', ':6.2f')
    top5 = hrutils.AverageMeter('Acc@5', ':6.2f')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = hrutils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)

            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def main(args):
    # TODO: code the loading of the model, dataset and training of the probes
    pass


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
    parser.add_argument('--RF_level2', default=3, type=int, help='Receptive field of model 2')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', default="cifar10", type=str, help='Dataset to use [cifar10,cifar100]')
    parser.add_argument('--model', default="resnet50", type=str, help='Architecture of model [resnet18,resnet50]')

    parser.add_argument('--name', '-n', default="no_name", help='name of the loss files, usually the seed name')
    parser.add_argument('--eval_set', '-es', default="val", help='On which set to performa the calculations')
    parser.add_argument('--folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where the output of the algorithm is going to be saved')
    parser.add_argument('--data_folder', default="/nobackup/sclaam/data", type=str,
                        help='Location of the dataset', required=True)
    parser.add_argument('--width', default=1, type=int, help='Width of the model')
    parser.add_argument('--eval_size', default=1000, type=int, help='How many images to use in the calculation')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size for loading data')
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
    parser.add_argument('--subtract_mean', default=1, type=int, help='Subtract mean of representations')

    args = parser.parse_args()

    if args.experiments == 1:
        main(args)
