import torch
from pathlib import Path
import time
import numpy as np
## FFCV imports
from typing import List
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, RandomResizedCrop, NormalizeImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from ffcv.transforms.common import Squeeze
import torchvision.transforms as torch_trnfs

class PytorchtoTensor(torch.nn.Module):
    def __init__(self, scale=1):
        super(PytorchtoTensor, self).__init__()
        self.scale = scale

    def forward(self, x):
        return torch_trnfs.ToTensor(x)
def make_ffcv_small_imagenet_dataloaders(train_dataset=None, val_dataset=None, batch_size=None, num_workers=2,
                                         distributed=False,
                                         in_memory=True, resolution=224, random_seed=None, valsize=5000, testsize=10000,
                                         shuffle_test=False, shuffle_val=True):
    if num_workers == 0:
        num_workers = 1

    length_small_imagenet = 99999

    start_time = time.time()

    # small_imagenet_MEAN_train= np.array([122.4760, 113.6542, 99.5722])
    # small_imagenet_STD_train = np.array( [69.5428, 66.8305, 70.2595])
    # small_imagenet_MEAN_test = np.array([120.6614, 112.3769, 98.3527])
    # small_imagenet_STD_test = np.array([68.9266, 66.2883, 69.4644])

    # CIFAR_STD = [51.5865, 50.847, 51.255]

    small_imagenet_MEAN_train = np.array([0.4802, 0.4481, 0.3975])
    small_imagenet_MEAN_test = np.array([0.4824, 0.4495, 0.3981])
    # #
    small_imagenet_STD_train = np.array([0.2302, 0.2265, 0.2262])
    small_imagenet_STD_test = np.array([0.2301, 0.2264, 0.2261])

    ########## train

    decoder = RandomResizedCropRGBImageDecoder((resolution, resolution))

    image_pipeline: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        PytorchtoTensor(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(small_imagenet_MEAN_train, small_imagenet_STD_train, np.float32)
        # NormalizeImage(np.array([0, 0, 0]), np.array([1, 1, 1]), np.float32)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"), non_blocking=True)
    ]

    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    if valsize == 0:
        train_loader = Loader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              # indices=whole_dataset_indices[:-valsize],
                              order=order,
                              os_cache=in_memory,
                              seed=random_seed,
                              drop_last=True,
                              pipelines={
                                  'image': image_pipeline,
                                  'label': label_pipeline
                              }, distributed=distributed)
    else:

        whole_dataset_indices = np.arange(length_small_imagenet)
        train_loader = Loader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              indices=whole_dataset_indices[:-valsize],
                              order=order,
                              os_cache=in_memory,
                              seed=random_seed,
                              drop_last=True,
                              pipelines={
                                  'image': image_pipeline,
                                  'label': label_pipeline
                              }, distributed=distributed)

    order = OrderOption.QUASI_RANDOM if shuffle_val else OrderOption.SEQUENTIAL
    if valsize != 0:
        val_loader = Loader(train_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            order=order,
                            indices=whole_dataset_indices[-valsize:],
                            os_cache=in_memory,
                            seed=random_seed,
                            drop_last=True,
                            pipelines={
                                'image': image_pipeline,
                                'label': label_pipeline
                            },
                            distributed=distributed)

        val_path = Path(val_dataset)
        assert val_path.is_file()
    else:
        val_loader = None

    res_tuple = (resolution, resolution)
    DEFAULT_CROP_RATIO = 224 / 256

    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline = [
        cropper,
        PytorchtoTensor(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(small_imagenet_MEAN_test, small_imagenet_STD_test, np.float32)
        # NormalizeImage(np.array([0, 0, 0]), np.array([1, 1, 1]), np.float32)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"),
                 non_blocking=True)
    ]
    test_order = OrderOption.SEQUENTIAL if not shuffle_test else OrderOption.QUASI_RANDOM
    if testsize < 10000:
        whole_testset_indices = np.arange(10000)
        test_loader = Loader(val_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             indices=np.random.choice(whole_testset_indices, testsize, replace=False),
                             order=test_order,
                             drop_last=False,
                             pipelines={
                                 'image': image_pipeline,
                                 'label': label_pipeline
                             },
                             distributed=distributed)
    else:
        test_loader = Loader(val_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             order=test_order,
                             drop_last=False,
                             pipelines={
                                 'image': image_pipeline,
                                 'label': label_pipeline
                             },
                             distributed=distributed)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    trainloader, valloader, testloader = make_ffcv_small_imagenet_dataloaders(
        "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv",
        "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv"
        ,
        512, 4, valsize=0, testsize=10000)
    mean = 0.0
    for images, _ in trainloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / 99999

    var = 0.0
    for images, _ in trainloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / torch.tensor(99999 * 224 * 224))
    print("Mean for train loader")
    print("{}".format(mean))
    print("STD for train loader")
    print("{}".format(std))

    mean = 0.0
    for images, _ in testloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / 10000

    var = 0.0
    for images, _ in testloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / torch.tensor(10000 * 224 * 224))

    print("\n Test loader ####################")
    print("Mean for test loader")
    print("{}".format(mean))
    print("STD for test loader")
    print("{}".format(std))
