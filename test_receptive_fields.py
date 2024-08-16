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
    size = [1, 3, 2000, 2000]
    le_rf = receptivefield(pytorch_dense_net, size)
    print(le_rf.rfsize)
    for block in blocks:
        print("Block {}".format(block))
        get_features_only_until_block_layer_densenet(net1, block=block, net_type=1)

        size = [1, 3, 4000, 4000]

        le_rf = receptivefield(net1, size)
        print(le_rf.rfsize)


if __name__ == '__main__':
    test_densenet_rf()
