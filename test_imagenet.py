import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.models import resnet18,ResNet18_Weights,resnet50,ResNet50_Weights
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
import time
# from ffcv.writer import DatasetWriter
# from ffcv.fields import RGBImageField, IntField
# from ffcv.loader import Loader, OrderOption
# from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
# from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder
# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct:torch.Tensor = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res













current_directory = Path().cwd()
data_path = ""
if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
    data_path = "/nobackup/sclaam/data/"
elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
    data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\MNIST"
elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
    data_path = "datasets/"
traindir = data_path+'imagenet/'+'train'
testdir=  data_path+'imagenet/'+'val'
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

train_dataset, val_dataset = torch.utils.data.random_split(whole_train_dataset, [1231167, 50000])

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
    train_dataset, batch_size=128, shuffle=True,
    num_workers=0, pin_memory=True, sampler=None)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=128, shuffle=True,
    num_workers=0, pin_memory=True, sampler=None)
test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=128, shuffle=False,
    num_workers=0, pin_memory=True)

net = resnet50()
net.load_state_dict(torch.load("/nobackup/sclaam/trained_models/resnet50_imagenet.pth"))

net.cuda()

net.train()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

end = time.time()

for batch_idx, (data, target) in enumerate(val_loader):
        data , target = data.cuda(), target.cuda()
        # switch to train mode
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        output = net(data)
        loss = criterion(output,target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                1, batch_idx, len(val_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

total_time = time.time() -end
hours_float = total_time/3600
decimal_part = hours_float - total_time//3600
minutes = decimal_part*60
print("Total time was {} hours and {} minutes".format(total_time//3600,minutes))
