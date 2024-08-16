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


class TruncatedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, fixed_points=None, classes=10):

        super(TruncatedBottleneck, self).__init__()

        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if fixed_points is None:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.fc1 = nn.Linear(planes, classes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.fc2 = nn.Linear(planes, classes)

            self.conv3 = nn.Conv2d(planes, self.expansion *
                                   planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)

            self.fc3 = nn.Linear(self.expansion * planes, classes)
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
        inter1 = self.avgpool(out)
        inter1 = inter1.view(inter1.size(0), -1)
        pred1 = self.fc1(inter1)
        out = self.relu(self.bn2(self.conv2(out)))

        inter2 = self.avgpool(out)
        inter2 = inter2.view(inter2.size(0), -1)
        pred2 = self.fc2(inter2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)

        inter3 = self.avgpool(out)
        inter3 = inter3.view(inter3.size(0), -1)
        pred3 = self.fc3(inter3)
        return [pred1, pred2, pred3], out


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
    def __init__(self, block: typing.Union[BasicBlock, Bottleneck], num_blocks, num_classes=10, multiplier=1,
                 fixed_points=None,
                 RF_level=1):
        super(ResNetRF, self).__init__()
        self.in_planes = 64 * multiplier
        self.fix_points = fixed_points
        self.rf_level = RF_level
        self.relu = nn.ReLU()
        self.width_multiplier = multiplier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.rf_level == 1:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        if self.rf_level == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.rf_level == 3:
            self.maxpool = nn.MaxPool2d(kernel_size=4, stride=3, padding=1)
        if self.rf_level == 4:
            self.maxpool = nn.MaxPool2d(kernel_size=5, stride=4, padding=1)
        if self.rf_level == 5:
            self.maxpool = nn.MaxPool2d(kernel_size=6, stride=5, padding=1)
        if self.rf_level == 6:
            self.maxpool = nn.MaxPool2d(kernel_size=7, stride=6, padding=1)
        if self.rf_level == 7:
            self.maxpool = nn.MaxPool2d(kernel_size=8, stride=7, padding=1)
        if self.rf_level == 8:
            self.maxpool = nn.MaxPool2d(kernel_size=9, stride=8, padding=1)
        if self.rf_level == 9:
            self.maxpool = nn.MaxPool2d(kernel_size=15, stride=14, padding=1)
        if self.rf_level == 10:
            self.maxpool = nn.MaxPool2d(kernel_size=20, stride=19, padding=1)
        if self.rf_level == 11:
            self.maxpool = nn.MaxPool2d(kernel_size=32, stride=31, padding=1)
        if self.fix_points is None:
            self.conv1 = nn.Conv2d(3, 64 * self.width_multiplier, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64 * self.width_multiplier)
            self.layer1 = self._make_layer(block, 64 * self.width_multiplier, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128 * self.width_multiplier, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256 * self.width_multiplier, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512 * self.width_multiplier, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion * self.width_multiplier, num_classes)
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
        # if self.rf_level != 1:
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # print("out:{}".format(out.size()))
        out = self.linear(out)
        return out


class TruncatedResNetRF(nn.Module):
    def __init__(self, block: typing.Union[BasicBlock, TruncatedBottleneck], num_blocks, num_classes=10,
                 multiplier=1,
                 fixed_points=None,
                 RF_level=1):
        super(TruncatedResNetRF, self).__init__()
        self.num_classes = num_classes
        self.in_planes = 64 * multiplier
        self.fix_points = fixed_points
        self.rf_level = RF_level
        self.relu = nn.ReLU()
        self.width_multiplier = multiplier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.rf_level == 0:
            self.maxpool = nn.Identity()
        if self.rf_level == 1:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        if self.rf_level == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.rf_level == 3:
            self.maxpool = nn.MaxPool2d(kernel_size=4, stride=3, padding=1)
        if self.rf_level == 4:
            self.maxpool = nn.MaxPool2d(kernel_size=5, stride=4, padding=1)
        if self.rf_level == 5:
            self.maxpool = nn.MaxPool2d(kernel_size=6, stride=5, padding=1)
        if self.rf_level == 6:
            self.maxpool = nn.MaxPool2d(kernel_size=7, stride=6, padding=1)
        if self.rf_level == 7:
            self.maxpool = nn.MaxPool2d(kernel_size=8, stride=7, padding=1)
        if self.rf_level == 8:
            self.maxpool = nn.MaxPool2d(kernel_size=9, stride=8, padding=1)
        if self.rf_level == 9:
            self.maxpool = nn.MaxPool2d(kernel_size=15, stride=14, padding=1)
        if self.rf_level == 10:
            self.maxpool = nn.MaxPool2d(kernel_size=20, stride=19, padding=1)
        if self.rf_level == 11:
            self.maxpool = nn.MaxPool2d(kernel_size=32, stride=31, padding=1)
        if self.fix_points is None:
            self.conv1 = nn.Conv2d(3, 64 * self.width_multiplier, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64 * self.width_multiplier)
            self.fc1 = nn.Linear(64, num_classes)
            self.layer1 = self._make_layer(block, 64 * self.width_multiplier, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128 * self.width_multiplier, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256 * self.width_multiplier, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512 * self.width_multiplier, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion * self.width_multiplier, num_classes)
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
            layers.append(block(self.in_planes, planes, stride, fixed_points=self.fix_points, classes=self.num_classes))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def set_fc_only_trainable(self):
        modules_true = []
        modules_false = []
        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                modules_false.append(module)
                for param in module.parameters():
                    param.requires_grad = False
            elif "linear" not in name and isinstance(module, nn.Linear):
                modules_true.append(module)
                for param in module.parameters():
                    param.requires_grad = True
        modules_false.append(self.linear)
        linear_params = self.linear.parameters()
        for p in linear_params:
            p.requires_grad = False
        return modules_true, modules_false

    def forward(self, x):
        intermediate_predictions = []
        out = self.relu(self.bn1(self.conv1(x)))
        inter1 = self.avgpool(out)
        inter1 = inter1.view(inter1.size(0), -1)
        pred1 = self.fc1(inter1)
        intermediate_predictions.append(pred1)
        # if self.rf_level != 1:
        out = self.maxpool(out)
        for m in self.layer1:
            inter_pred, out = m(out)
            intermediate_predictions.extend(inter_pred)
        for m in self.layer2:
            inter_pred, out = m(out)
            intermediate_predictions.extend(inter_pred)
        for m in self.layer3:
            inter_pred, out = m(out)
            intermediate_predictions.extend(inter_pred)
        for m in self.layer4:
            inter_pred, out = m(out)
            intermediate_predictions.extend(inter_pred)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # print("out:{}".format(out.size()))
        out = self.linear(out)
        return intermediate_predictions, out


def ResNet18(num_classes=10, fix_points=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, fix_points)


def ResNet18_rf(num_classes=10, fix_points=None, RF_level=1, multiplier=1):
    rf_level = RF_level
    if rf_level == 0:
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, fix_points)
    else:
        return ResNetRF(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, fixed_points=fix_points, RF_level=rf_level,
                        multiplier=multiplier)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(num_classes=10, fix_points=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, fix_points)


def ResNet50_rf(num_classes=10, fix_points=None, rf_level=1, multiplier=1):
    if rf_level == 0:
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, fix_points)
    else:
        return ResNetRF(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, fixed_points=fix_points, RF_level=rf_level,
                        multiplier=multiplier)


def truncated_ResNet50_rf(num_classes=10, fix_points=None, rf_level=1, multiplier=1):
    if rf_level == 0:
        return TruncatedResNetRF(TruncatedBottleneck, [3, 4, 6, 3], num_classes, fix_points)
    else:
        return TruncatedResNetRF(TruncatedBottleneck, [3, 4, 6, 3], num_classes=num_classes, fixed_points=fix_points,
                                 RF_level=rf_level,
                                 multiplier=multiplier)


def ResNet24_rf(num_classes=10, fix_points=None, rf_level=1, multiplier=1):
    if rf_level == 0:
        return ResNet(Bottleneck, [2, 2, 2, 2], num_classes, fix_points)
    else:
        return ResNetRF(Bottleneck, [2, 2, 2, 2], num_classes=num_classes, fixed_points=fix_points, RF_level=rf_level,
                        multiplier=multiplier)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    from easy_receptive_fields_pytorch.receptivefield import receptivefield, give_effective_receptive_field
    from main import get_features_only_until_block_layer
    from torchvision.models import resnet18, resnet50
    import matplotlib.pyplot as plt

    print("Receptive field normal resnet18")

    # net = resnet18(pretrained=False)

    # net.fc = torch.nn.Linear(512, 10)

    # net.to(device)

    # y = net(torch.randn(3, 3, 32, 32))
    # print(y)
    blocks = [4]
    receptive_fields = ["k6", "k7", "k8", "k9", 5, 6, 7]

    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    for rf in receptive_fields:

        net = ResNet18_rf(num_classes=10, RF_level=rf)

        y = net(torch.randn(3, 3, 32, 32))

        print("######################")
        print(rf)
        print("######################")

        get_features_only_until_block_layer(net, block=4, net_type=1)

        size = [1, 3, 1000, 1000]

        if rf == 5:
            size = [1, 3, 1500, 1500]

        if rf == 6:
            size = [1, 3, 2000, 2000]

        if rf == 7:
            size = [1, 3, 3100, 3100]

        le_rf = receptivefield(net, size)
        # y = net(torch.randn(3, 3, 32, 32))
        print(le_rf.rfsize)
        # samples = []
        # for j in range(10):
        #     net = ResNet50_rf(num_classes=10, rf_level=rf)
        #     get_features_only_until_block_layer(net, block=4, net_type=1)
        #
        #     in_grad, input_pos = give_effective_receptive_field(net, (1, 3, 32, 32))
        #
        #     samples.append(torch.abs(in_grad))
        #
        # stacked_samples = torch.stack(samples, dim=0)
        # mean = stacked_samples.mean(dim=0)

        # plt.vlines(x=input_pos[0] - 16, ymin=input_pos[1] - 16, ymax=input_pos[1] + 16, color='r')
        # plt.vlines(x=input_pos[0] + 16, ymin=input_pos[1] - 16, ymax=input_pos[1] + 16, color='r')
        # plt.hlines(y=input_pos[1] - 16, xmin=input_pos[0] - 16, xmax=input_pos[0] + 16, color='r')
        # plt.hlines(y=input_pos[1] + 16, xmin=input_pos[0] - 16, xmax=input_pos[0] + 16, color='r')
        # axs[rf-1].matshow(mean, cmap="magma")
        # axs[rf-1].set_title("RF {}".format(rf))

    # plt.tight_layout()
    # plt.show()

    # pdb.set_trace()

    # receptive_fields.append(tuple(rf.rfsize))

    # print(y.size())


# test()
if __name__ == '__main__':
    test()
