'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)


def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32)


def DenseNet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32)


def DenseNet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)


def densenet_cifar():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12)


def test():
    net = densenet_cifar()
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y)


def get_features_only_until_block_layer_densenet_pytorch(net, block=2, net_type=0):
    def features_only(self, x):
        out = self.features(x)
        return out

    net.forward = features_only.__get__(net)  # bind method


def get_features_only_until_block_layer_densenet(net, block=2, net_type=0):
    # ResNet block to compute receptive field for

    def features_only(self, x):
        out = self.conv1(x)
        if block == 0: return out
        out = self.trans1(self.dense1(out))
        if block == 1: return out
        out = self.trans2(self.dense2(out))
        if block == 2: return out
        out = self.trans3(self.dense3(out))
        if block == 3: return out
        out = self.dense4(out)
        if block == 4: return out
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    net.forward = features_only.__get__(net)  # bind method

def get_features_only_until_block_layer_densenet_corin(net, block=2, net_type=0):
    # ResNet block to compute receptive field for

    def features_only(self, x):
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

    net.forward = features_only.__get__(net)  # bind method

def test_rf():
    from easy_receptive_fields_pytorch.receptivefield import receptivefield, give_effective_receptive_field
    from main import get_features_only_until_block_layer
    from torchvision.models import densenet121
    from densenet_corin import densenet_40
    pytorch_dense_net = densenet121()
    net1 = densenet_40([0]*100)
    net2 = DenseNet121()

    # save_onnx(pytorch_dense_net, "pytorch_densenet_121")
    # save_onnx(net1, "densenet_cifar")
    # save_onnx(net2, "local_densenet_121")
    blocks = [0, 1, 2, 3, 4]
    # print("RF of pytorch implementation")
    # get_features_only_until_block_layer_densenet_pytorch(pytorch_dense_net)
    # size = [1, 3, 300, 2000]
    # le_rf = receptivefield(pytorch_dense_net, size)
    # print(le_rf.rfsize)
    # for block in blocks:
    # print("Block {}".format(block))
    get_features_only_until_block_layer_densenet_corin(net1, block=4, net_type=1)
    size = [1, 3, 600, 600]
    le_rf = receptivefield(net1, size)
    print(le_rf.rfsize)


def save_onnx(net, name: str = ""):
    input_names = ['Image']

    output_names = ['y_hat']

    size = [2, 3, 32, 32]
    dummy_input = torch.randn(size)
    torch.onnx.export(net, dummy_input,
                      f'/home/luisaam/PycharmProjects/sparse_ensemble/onnx_models/onnx_models_{name}.onnx',
                      input_names=input_names,
                      output_names=output_names)
    # y = net(torch.randn(3, 3, 32, 32))


if __name__ == '__main__':
    test_rf()

# test()
