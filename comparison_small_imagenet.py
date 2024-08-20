from test_imagenet import load_small_imagenet
from ffcv_loaders import make_ffcv_small_imagenet_dataloaders
import PIL
from pathlib import Path
import omegaconf
import numpy as np
import torchvision.transforms.functional as F


def test_images(args):
    current_directory = Path().cwd()
    batch_size = 10
    data_path = "."
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        data_path = "/nobackup/sclaam/data"
    elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
        data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
    elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
        data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
    elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
        data_path = "/home/luisaam/Documents/PhD/data/"

    from ffcv_loaders import make_ffcv_small_imagenet_dataloaders
    ffcv_trainloader, ffcv_valloader, ffcv_testloader = make_ffcv_small_imagenet_dataloaders(args.ffcv_train,
                                                                                             args.ffcv_val,
                                                                                             batch_size,
                                                                                             args.num_workers,
                                                                                             valsize=5000,
                                                                                             testsize=10000,
                                                                                             shuffle_val=True,
                                                                                             shuffle_test=False, )

    from test_imagenet import load_small_imagenet
    trainloader, valloader, testloader = load_small_imagenet(
        {"traindir": data_path + "/small_imagenet/train", "valdir": data_path + "/small_imagenet/val",
         "num_workers": args.num_workers, "batch_size": batch_size, "resolution": args.input_resolution},
        val_size=5000, test_size=10000, shuffle_val=True, shuffle_test=False)
    for (imgs1, y1), (imgs_ffcv, y2) in zip(testloader, ffcv_testloader):
        print(f"Normal loader labels {y1}")
        print(f"FCCV loader labels {y2}")
        i = 0
        for img1 in imgs1:
            pil_image = F.to_pil_image(img1)
            pil_image.save(f"{i}_normal.png")
            i += 1
        i = 0
        for img2 in imgs_ffcv:
            pil_image = F.to_pil_image(img2)
            pil_image.save(f"{i}_normal.png")
            i += 1
        np.save(imgs1.numpy(), "normal_images.npy")
        np.save(imgs_ffcv.numpy(), "ffcv_images.pny")
        break


def main():
    cfg = omegaconf.DictConfig({
        # "solution": "/home/luisaam/checkpoints/resnet_small_normal_small_imagenet_seed.8_rf_level_5_recording_200_test_acc_62.13.pth",
        # "solution": "/home/luisaam/checkpoints/vgg19_normal_cifar10_1723720946.9104598_rf_level_1_recording_200_no_ffcv_test_acc_93.77.pth",
        "solution": "/home/luisaam/checkpoints/resnet50_normal_cifar10_1723722961.8540442_rf_level_2_recording_200_no_ffcv_test_acc_94.24.pth",
        "modeltype1": "normal",
        "seedname1": "_seed_8",
        "RF_level": 2,
        "epochs": 1,
        "ffcv": 0,
        "ffcv_val": "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv",
        "ffcv_train": "/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv",
        "batch_size": 64,
        "model": "resnet50",
        "dataset": "cifar10",
        "num_workers": 0,
        "input_resolution": 32,
        "width": 1,
        "name": "no_name",
        "job_dir": "truncated_results_local",
        "lr": 0.1,
        "resume": False,
        "eval_size": 5000,

    })

    test_images(cfg)


if __name__ == '__main__':
    main()
