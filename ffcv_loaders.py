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

def make_ffcv_small_imagenet_dataloaders(train_dataset=None, val_dataset=None, batch_size=None, num_workers=2,
                                         distributed=False,
                                         in_memory=True):
    paths = {
        'train': train_dataset,
        'test': val_dataset

    }

    start_time = time.time()
    small_imagenet_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    small_imagenet_MEAN = np.array([0.4802, 0.4481, 0.3975])
    small_imagenet_STD = np.array([0.2302, 0.2265, 0.2262])
    loaders = {}

    # for name in ['train', 'test']:
    ########## train
    decoder = RandomResizedCropRGBImageDecoder((360, 360))
    image_pipeline: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(small_imagenet_MEAN, small_imagenet_STD,np.float32)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"), non_blocking=True)
    ]

    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    train_loader = Loader(train_dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          order=order,
                          os_cache=in_memory,
                          drop_last=True,
                          pipelines={
                              'image': image_pipeline,
                              'label': label_pipeline
                          },
                          distributed=distributed)

    val_path = Path(val_dataset)
    assert val_path.is_file()
    res_tuple = (360, 360)
    DEFAULT_CROP_RATIO = 360 / 392
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(small_imagenet_MEAN, small_imagenet_STD, np.float32)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"),
                 non_blocking=True)
    ]

    test_loader = Loader(val_dataset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         order=OrderOption.SEQUENTIAL,
                         drop_last=False,
                         pipelines={
                             'image': image_pipeline,
                             'label': label_pipeline
                         },
                         distributed=distributed)

    return train_loader, None, test_loader

