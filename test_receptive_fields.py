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
    from alternate_models.small_models import deep_small_ResNet_rf,get_output_until_block_deep_small_resnet

    # blocks = [3, 4, 5, 6, 7, 8, 9, 10]
    blocks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    for i in blocks:
        # samples = []
        # for j in range(10):
        x = torch.randn(3, 3, 400, 400)
        dummy_y = torch.zeros(3, 10)
        dummy_y[:, 5] = 1
        #
        # if i < 5:
        # vgg_net = small_VGG_RF('small_vgg', RF_level=i)
        # #     y_vgg = vgg_net(x)
        # #     print(y_vgg)
        # get_output_until_block_small_vgg(vgg_net, block=4, net_type=1)
        # vgg_rf = receptivefield(vgg_net, (1, 3, 1500, 1500))
        # print("Receptive field of Small Vgg Level {}".format(i))
        # print(vgg_rf.rfsize)
        #
        #     print("Receptive field of VGG")
        #     print(vgg_rf)

        resnet_net = deep_small_ResNet_rf(10, RF_level=i)

        # y_resnet = resnet_net(x)
        # input_names = ['Image']
        #
        # output_names = ['y_hat']
        #
        # torch.onnx.export(resnet_net, x,
        #                   f'onnx_model_small_resnet_small_imagenet.onnx',
        #                   input_names=input_names,
        #                   output_names=output_names)
        #
        # loss = dummy_loss(y_resnet, dummy_y)
        # loss.backward()
        # print(y_resnet)

        get_output_until_block_deep_small_resnet(resnet_net, block=4, net_type=1)
        resnet_rf = receptivefield(resnet_net, (1, 3, 2000, 2000))
        print("Receptive field of deep small ResNet Level {}".format(i))
        print(resnet_rf.rfsize)

if __name__ == '__main__':
    test_deep_RF_models()