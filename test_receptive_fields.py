import traceback

import torch

from easy_receptive_fields_pytorch.receptivefield import receptivefield, give_effective_receptive_field
from alternate_models.densenet import test_rf, get_features_only_until_block_layer_densenet, \
    get_features_only_until_block_layer_densenet_pytorch, DenseNet121, densenet_cifar


def test_densenet_rf():
    from easy_receptive_fields_pytorch.receptivefield import receptivefield, give_effective_receptive_field
    from main import get_features_only_until_block_layer
    from torchvision.models import densenet121
    pytorch_dense_net = densenet121()
    net1 = densenet_cifar()
    net2 = DenseNet121()
    #
    # save_onnx(pytorch_dense_net,"pytorch_densenet_121")
    # save_onnx(net1,"densenet_cifar")
    # save_onnx(net2,"local_densenet_121")
    blocks = [0, 1, 2, 3, 4]
    print("RF of pytorch implementation")
    get_features_only_until_block_layer_densenet_pytorch(pytorch_dense_net)
    size = [1, 3, 4000, 4000]

    try:
        le_rf = receptivefield(pytorch_dense_net, size)
        print(le_rf.rfsize)
        print("Receptive field:\n{}".format(le_rf.rfsize))
    except Exception as e:
        print("****************")
        print(e)
        print("Receptive field is grater than {}".format(4000))

    print("RF of local implementation")
    for block in blocks:

        print("Block {}".format(block))
        get_features_only_until_block_layer_densenet(net1, block=block, net_type=1)

        size = [1, 3, 4000, 4000]

        try:
            le_rf = receptivefield(net1, size)
            print(le_rf.rfsize)
            print("Receptive field:\n{}".format(le_rf.rfsize))
        except Exception as e:
            print("****************")
            print(e)
            print("Receptive field is grater than {}".format(4000))


def test_deep_RF_models():
    from easy_receptive_fields_pytorch.receptivefield import receptivefield, give_effective_receptive_field
    import torch
    from alternate_models.small_models import deep_small_ResNet_rf, get_output_until_block_deep_small_resnet

    # blocks = [3, 4, 5, 6, 7, 8, 9, 10]
    blocks = [10, 11]
    # blocks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    for i in blocks:
        resnet_net = deep_small_ResNet_rf(10, RF_level=i)
        get_output_until_block_deep_small_resnet(resnet_net, block=4, net_type=1)
        resnet_rf = receptivefield(resnet_net, (1, 3, 3000, 3000))
        print("Receptive field of deep small ResNet Level {}".format(i))
        print(resnet_rf.rfsize)


def test_RF_densenet40():
    from easy_receptive_fields_pytorch.receptivefield import receptivefield
    from alternate_models.densenet_corin import densenet_40_RF
    from alternate_models.densenet import get_features_only_until_block_layer_densenet_corin

    blocks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    for i in blocks:
        try:
            net1 = densenet_40_RF([0] * 100, RF_level=i)
            print("Receptive field of Densenet40 Level {}".format(i))
            get_features_only_until_block_layer_densenet_corin(net1, block=4, net_type=1)
            size = [1, 3, 900, 900]
            le_rf = receptivefield(net1, size)
            print(le_rf.rfsize)
        except Exception as e:
            print(traceback.format_exc())
            print("The receptive field for level {} of densenet40 is greater than 5000".format(i))


def test_RF_mobilenet_cifar():
    from easy_receptive_fields_pytorch.receptivefield import receptivefield
    from alternate_models.mobilenetv2 import get_features_mobilenetv2, MobileNetV2_cifar_RF

    blocks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    for i in blocks:
        try:
            print("Receptive field of MobileNet cifar Level {}".format(i))
            net1 = MobileNetV2_cifar_RF(num_classes=10, RF_level=i)
            get_features_mobilenetv2(net1)
            size = [1, 3, 5000, 5000]
            le_rf = receptivefield(net1, size)
            print(le_rf.rfsize)
        except Exception:
            print(traceback.format_exc())
            print("The receptive field for level {} in Mobile net cifar is greater than 5000".format(i))


def test_RF_mobilenet_imagenet():
    from easy_receptive_fields_pytorch.receptivefield import receptivefield
    from alternate_models.mobilenetv2 import get_features_mobilenetv2, MobileNetV2_imagenet_RF

    blocks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    for i in blocks:
        try:
            print("Receptive field of MobileNet imagenet Level {}".format(i))
            net1 = MobileNetV2_imagenet_RF(num_classes=10, RF_level=i)
            get_features_mobilenetv2(net1)
            size = [1, 3, 6000, 6000]
            le_rf = receptivefield(net1, size)
            print(le_rf.rfsize)
        except Exception as e:
            print(traceback.format_exc())
            print("The receptive field for level {}  in Mobilenet imagenet is greater than 5000".format(i))


def test_RF_resnet50_stride():
    from easy_receptive_fields_pytorch.receptivefield import receptivefield
    from alternate_models.resnet import get_resnet_features, ResNet50_rf_stride

    blocks = [9, 10, 11]

    for i in blocks:
        try:
            print("Receptive field of RSENET50 with stride Level {}".format(i))
            net1 = ResNet50_rf_stride(num_classes=10, rf_level=i)
            get_resnet_features(net1)
            size = [1, 3, 1500, 1500]
            le_rf = receptivefield(net1, size)
            print(le_rf.rfsize)
        except Exception as e:
            print(traceback.format_exc())
            print("The receptive field for level {}  in resnet50 stride is greater than 5000".format(i))


def test_RF_vgg_stride():
    from easy_receptive_fields_pytorch.receptivefield import receptivefield
    from alternate_models.vgg import VGG_RF_stride, get_features_only_VGG

    blocks = [11]

    for i in blocks:
        try:
            print("Receptive field of vgg19 with stride Level {}".format(i))
            net1 = VGG_RF_stride(vgg_name="VGG19_rf", num_classes=10, RF_level=i)
            get_features_only_VGG(net1)
            size = [1, 3, 2500, 2500]
            le_rf = receptivefield(net1, size)
            print(le_rf.rfsize)
            x = torch.rand(2, 3, 32, 32)
            # net1(x)
            print("all good for cifar10")
        except Exception as e:
            print(traceback.format_exc())
            print("The receptive field for level {}  in VGG19 stride is greater than 5000".format(i))


if __name__ == '__main__':
    # test_deep_RF_models()
    test_RF_densenet40()
    test_RF_mobilenet_cifar()
    # test_RF_resnet50_stride()
    # test_RF_vgg_stride()
