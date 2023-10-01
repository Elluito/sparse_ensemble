'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import dnn_mode_connectivity.curves as curves


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, fixed_points=None):
        super(BasicBlock, self).__init__()

        self.relu = nn.ReLU()
        if fixed_points is None:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        if fixed_points is not None:

            self.conv1 = curves.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, fix_points=fixed_points)
            self.bn1 = curves.BatchNorm2d(planes, fix_points=fixed_points)
            self.conv2 = curves.Conv2d(planes, planes, kernel_size=3,
                                       stride=1, padding=1, bias=False, fix_points=fixed_points)
            self.bn2 = curves.BatchNorm2d(planes, fix_points=fixed_points)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    curves.Conv2d(in_planes, self.expansion * planes,
                                  kernel_size=1, stride=stride, bias=False, fix_points=fixed_points),
                    curves.BatchNorm2d(self.expansion * planes, fix_points=fixed_points)
                )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, fixed_points=None):
        super(Bottleneck, self).__init__()

        self.relu = nn.ReLU()
        if fixed_points is None:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion *
                                   planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        if fixed_points is not None:

            self.conv1 = curves.Conv2d(in_planes, planes, kernel_size=1, bias=False, fix_points=fixed_points)
            self.bn1 = curves.BatchNorm2d(planes, fix_points=fixed_points)
            self.conv2 = curves.Conv2d(planes, planes, kernel_size=3,
                                       stride=stride, padding=1, bias=False, fix_points=fixed_points)
            self.bn2 = curves.BatchNorm2d(planes, fix_points=fixed_points)
            self.conv3 = curves.Conv2d(planes, self.expansion *
                                       planes, kernel_size=1, bias=False, fix_points=fixed_points)
            self.bn3 = curves.BatchNorm2d(self.expansion * planes, fix_points=fixed_points)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    curves.Conv2d(in_planes, self.expansion * planes,
                                  kernel_size=1, stride=stride, bias=False, fix_points=fixed_points),
                    curves.BatchNorm2d(self.expansion * planes, fix_points=fixed_points)
                )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: typing.Union[BasicBlock, Bottleneck], num_blocks, num_classes=10, fixed_points=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.fix_points = fixed_points
        self.relu = nn.ReLU()
        if self.fix_points is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        if self.fix_points is not None:
            self.conv1 = curves.Conv2d(3, 64, kernel_size=3,
                                       stride=1, padding=1, bias=False, fix_points=self.fix_points)
            self.bn1 = curves.BatchNorm2d(64, fix_points=self.fix_points)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = curves.Linear(512 * block.expansion, num_classes, self.fix_points)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, fixed_points=self.fix_points))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetCurve(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, fixed_points=None):
        super(ResNetCurve, self).__init__()
        self.in_planes = 64
        self.fix_points = fixed_points
        self.conv1 = curves.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False, fix_points=self.fix_points)
        self.bn1 = curves.BatchNorm2d(64, fix_points=self.fix_points)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = curves.Linear(512 * block.expansion, num_classes, self.fix_points)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, fixed_points=self.fix_points))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetRF(nn.Module):
    def __init__(self, block: typing.Union[BasicBlock, Bottleneck], num_blocks, num_classes=10, fixed_points=None,
                 RF_level=1):
        super(ResNetRF, self).__init__()
        self.in_planes = 64
        self.fix_points = fixed_points
        self.rf_level = RF_level
        self.relu = nn.ReLU()
        if self.rf_level == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        if self.rf_level == 3:
            self.maxpool = nn.MaxPool2d(kernel_size=4, stride=3, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        if self.rf_level == 4:
            self.maxpool = nn.MaxPool2d(kernel_size=5, stride=4, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.fix_points is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        if self.fix_points is not None:
            self.conv1 = curves.Conv2d(3, 64, kernel_size=3,
                                       stride=1, padding=1, bias=False, fix_points=self.fix_points)
            self.bn1 = curves.BatchNorm2d(64, fix_points=self.fix_points)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = curves.Linear(512 * block.expansion, num_classes, self.fix_points)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, fixed_points=self.fix_points))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if self.rf_level != 1:
            out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10, fix_points=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, fix_points)


def ResNet18_rf(num_classes=10, fix_points=None, rf_level=1):
    return ResNetRF(BasicBlock, [2, 2, 2, 2], num_classes, fix_points, RF_level=rf_level)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(num_classes=10, fix_points=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, fix_points)


def ResNet50_rf(num_classes=10, fix_points=None, rf_level=1):
    return ResNetRF(Bottleneck, [3, 4, 6, 3], num_classes, fix_points, RF_level=rf_level)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()
if __name__ == '__main__':
    #
    # from easy_receptive_fields_pytorch.receptivefield import receptivefield
    # from main import get_features_only_until_block_layer
    # import pdb
    #
    # # resnet18_normal.cpu()
    # # resnet18_normal.train()
    # # receptive_field_dict = receptivefield(resnet18_normal, (3, 32, 32),device="cpu")
    # # get_features_only_until_layer(resnet18_pytorch,block=1,net_type=0)
    # # rf = receptivefield(resnet18_pytorch, (1, 3, 224, 224))
    # # print(rf)
    # print("Receptive field normal resnet18")
    # net = ResNet50_rf(num_classes=10, rf_level=2)
    #
    # # resnet18_normal.train()
    # # get_features_only_until_layer(resnet18_normal, block=0.75, net_type=1)
    # # rf = receptivefield(resnet18_normal, (1, 3, 224, 224))
    # # print(rf)
    # # blocks = [0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4]
    # # blocks = np.linspace(0,4,17)
    # blocks = [0, 1, 2, 3, 4]
    # receptive_fields = []
    #
    # for i in blocks:
    #     get_features_only_until_block_layer(net, block=i, net_type=1)
    #     rf = receptivefield(net, (1, 3, 500, 500))
    #     # pdb.set_trace()
    #     print("Receptive field for block {}".format(i))
    #     print(rf)
    #     receptive_fields.append(tuple(rf.rfsize))
    #     # rf.show()
    #
    #
    #
    # y = net(torch.randn(3, 3, 32, 32))
    # print(y.size())
