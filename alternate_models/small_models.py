import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import dnn_mode_connectivity.curves as curves
import traceback

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_rf': [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, 512, 512, 512, 512],
    'VGG19_rf_mod': [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, 512, 512, 512, 512],
    "small_vgg": [64, 64, "M", 128, 128, "M", 256],
    "deep_small_vgg": [64, 64, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512],
    "deep_small_vgg_2": [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 512, 512, 512, 512]
}


class small_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, fixed_points=None):
        super(small_BasicBlock, self).__init__()

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


class small_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, fixed_points=None):
        super(small_Bottleneck, self).__init__()

        self.relu = nn.ReLU()
        if fixed_points is None:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, self.expansion * planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.expansion * planes)
            # self.conv3 = nn.Conv2d(planes, self.expansion *
            #                        planes, kernel_size=1, bias=False)
            # self.bn3 = nn.BatchNorm2d(self.expansion * planes)

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
        out = self.bn2(self.conv2(out))
        # out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class small_deep_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, fixed_points=None):
        super(small_deep_Bottleneck, self).__init__()

        self.relu = nn.ReLU()
        if fixed_points is None:
            self.conv1_1x1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2_1x1 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3_3x3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes)
            self.conv4_1x1 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm2d(planes)

            self.conv5_1x1 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
            self.bn5 = nn.BatchNorm2d(self.expansion * planes)
            # self.conv3 = nn.Conv2d(planes, self.expansion *
            #                        planes, kernel_size=1, bias=False)
            # self.bn3 = nn.BatchNorm2d(self.expansion * planes)

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
        out = self.relu(self.bn1(self.conv1_1x1(x)))
        out = self.relu(self.bn2(self.conv2_1x1(out)))
        out = self.relu(self.bn3(self.conv3_3x3(out)))
        out = self.relu(self.bn4(self.conv4_1x1(out)))
        out = self.bn5(self.conv5_1x1(out))
        # out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: typing.Union[small_BasicBlock, small_Bottleneck], num_blocks, num_classes=10,
                 fixed_points=None):
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


class small_ResNetRF(nn.Module):
    def __init__(self, block: typing.Union[small_BasicBlock, small_Bottleneck], num_blocks, num_classes=10,
                 multiplier=1,
                 fixed_points=None,
                 RF_level=1):
        super(small_ResNetRF, self).__init__()
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
            self.maxpool = nn.MaxPool2d(kernel_size=15, stride=14, padding=1)
        if self.rf_level == 6:
            self.maxpool = nn.MaxPool2d(kernel_size=20, stride=19, padding=1)
        if self.rf_level == 7:
            self.maxpool = nn.MaxPool2d(kernel_size=35, stride=34, padding=1)
        if self.rf_level == 8:
            self.maxpool = nn.MaxPool2d(kernel_size=45, stride=44, padding=1)
        if self.rf_level == 9:
            self.maxpool = nn.MaxPool2d(kernel_size=55, stride=54, padding=1)
        if self.rf_level == 10:
            self.maxpool = nn.MaxPool2d(kernel_size=64, stride=63, padding=1)
        # if self.fix_points is None:
        #     self.conv1 = nn.Conv2d(3, 64 * self.width_multiplier, kernel_size=3,
        #                            stride=1, padding=1, bias=False)
        #     self.bn1 = nn.BatchNorm2d(64 * self.width_multiplier)
        #     self.layer1 = self._make_layer(block, 64 * self.width_multiplier, num_blocks[0], stride=1)
        #     self.layer2 = self._make_layer(block, 128 * self.width_multiplier, num_blocks[1], stride=2)
        #     self.layer3 = self._make_layer(block, 256 * self.width_multiplier, num_blocks[2], stride=2)
        #     self.layer4 = self._make_layer(block, 512 * self.width_multiplier, num_blocks[3], stride=2)
        #     self.linear = nn.Linear(512 * block.expansion * self.width_multiplier, num_classes)
        # if self.fix_points is not None:
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        # conv2 = nn.Conv2d(128, 1 * 256, kernel_size=3,
        #                   stride=2, padding=1, bias=False)
        # bn2 = nn.BatchNorm2d(1 * 256)
        # self.layer3 = nn.Sequential(*[conv2, bn2])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(256 * block.expansion, num_classes)

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
        # print("shape layer1: {}".format(out.shape))
        out = self.layer2(out)
        # print("shape layer2: {}".format(out.shape))
        out = self.layer3(out)
        # print("shape layer3: {}".format(out.shape))
        # out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # print("out:{}".format(out.size()))
        out = self.linear(out)
        return out


class Deep_small_ResNetRF(nn.Module):
    def __init__(self, block: typing.Union[small_BasicBlock, small_Bottleneck], num_blocks, num_classes=10,
                 multiplier=1,
                 fixed_points=None,
                 RF_level=1):
        super(Deep_small_ResNetRF, self).__init__()

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
            # self.maxpool = nn.MaxPool2d(kernel_size=15, stride=14, padding=1)
            self.maxpool = nn.MaxPool2d(kernel_size=6, stride=5, padding=1)
        if self.rf_level == 6:
            self.maxpool = nn.MaxPool2d(kernel_size=7, stride=6, padding=1)
            # self.maxpool = nn.MaxPool2d(kernel_size=20, stride=19, padding=1)
        if self.rf_level == 7:
            self.maxpool = nn.MaxPool2d(kernel_size=8, stride=7, padding=1)
            # self.maxpool = nn.MaxPool2d(kernel_size=35, stride=34, padding=1)
        if self.rf_level == 8:
            self.maxpool = nn.MaxPool2d(kernel_size=9, stride=8, padding=1)
            # self.maxpool = nn.MaxPool2d(kernel_size=45, stride=44, padding=1)
        if self.rf_level == 9:
            self.maxpool = nn.MaxPool2d(kernel_size=10, stride=9, padding=1)
            # self.maxpool = nn.MaxPool2d(kernel_size=55, stride=54, padding=1)
        if self.rf_level == 10:
            self.maxpool = nn.MaxPool2d(kernel_size=11, stride=10, padding=1)
        if self.rf_level == 11:
            self.maxpool = nn.MaxPool2d(kernel_size=44, stride=43, padding=1)
        if self.rf_level == 12:
            self.maxpool = nn.MaxPool2d(kernel_size=58, stride=57, padding=1)
        if self.rf_level == 13:
            self.maxpool = nn.MaxPool2d(kernel_size=94, stride=93, padding=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=64, stride=63, padding=1)
        # if self.fix_points is None:
        #     self.conv1 = nn.Conv2d(3, 64 * self.width_multiplier, kernel_size=3,
        #                            stride=1, padding=1, bias=False)
        #     self.bn1 = nn.BatchNorm2d(64 * self.width_multiplier)
        #     self.layer1 = self._make_layer(block, 64 * self.width_multiplier, num_blocks[0], stride=1)
        #     self.layer2 = self._make_layer(block, 128 * self.width_multiplier, num_blocks[1], stride=2)
        #     self.layer3 = self._make_layer(block, 256 * self.width_multiplier, num_blocks[2], stride=2)
        #     self.layer4 = self._make_layer(block, 512 * self.width_multiplier, num_blocks[3], stride=2)
        #     self.linear = nn.Linear(512 * block.expansion * self.width_multiplier, num_classes)
        # if self.fix_points is not None:
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        # conv2 = nn.Conv2d(128, 1 * 256, kernel_size=3,
        #                   stride=2, padding=1, bias=False)
        # bn2 = nn.BatchNorm2d(1 * 256)
        # self.layer3 = nn.Sequential(*[conv2, bn2])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

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
        # print("shape layer1: {}".format(out.shape))
        out = self.layer2(out)
        # print("shape layer2: {}".format(out.shape))
        out = self.layer3(out)
        # print("shape layer3: {}".format(out.shape))
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # print("out:{}".format(out.size()))
        out = self.linear(out)
        return out


class small_VGG_RF(nn.Module):
    def __init__(self, vgg_name, num_classes=10, RF_level=0):
        super(small_VGG_RF, self).__init__()
        self.rf_level = RF_level
        self.maxpool = None
        self.config = cfg[vgg_name]
        if self.rf_level == 1:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
            self.config = cfg[vgg_name]
        if self.rf_level == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.config = cfg[vgg_name]
        if self.rf_level == 3:
            self.maxpool = nn.MaxPool2d(kernel_size=4, stride=3, padding=1)
            self.config = cfg[vgg_name]
        if self.rf_level == 4:
            self.maxpool = nn.MaxPool2d(kernel_size=5, stride=4, padding=1)
            self.config = cfg[vgg_name]
        if self.rf_level == 5:
            self.maxpool = nn.MaxPool2d(kernel_size=15, stride=14, padding=1)
        if self.rf_level == 6:
            self.maxpool = nn.MaxPool2d(kernel_size=20, stride=19, padding=1)
        if self.rf_level == 7:
            self.maxpool = nn.MaxPool2d(kernel_size=32, stride=31, padding=1)
        if self.rf_level == 8:
            self.maxpool = nn.MaxPool2d(kernel_size=45, stride=44, padding=1)
        if self.rf_level == 9:
            self.maxpool = nn.MaxPool2d(kernel_size=55, stride=54, padding=1)
        if self.rf_level == 10:
            self.maxpool = nn.MaxPool2d(kernel_size=64, stride=63, padding=1)

        self.features = self._make_layers(self.config)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):

        # for i, layer in enumerate(self.features):
        #     x = layer(x)
        #     try:
        #         print("{}".format(cfg["VGG19"][i]))
        #         print("output shape of layer {}/{}: {}".format(i, len(self.features), x.size()))
        #     except:
        #
        #         print("output shape of layer {}/{}: {}".format(i, len(self.features), x.size()))
        out = self.features(x)
        # out = x
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        if self.maxpool:
            layers.insert(1, self.maxpool)
        return nn.Sequential(*layers)


class DeepSmallVGG_RF(nn.Module):
    def __init__(self, vgg_name, num_classes=10, RF_level=0):
        super(DeepSmallVGG_RF, self).__init__()
        self.rf_level = RF_level
        self.maxpool = None
        self.config = cfg[vgg_name]
        if self.rf_level == 1:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
            self.config = cfg[vgg_name]
        if self.rf_level == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.config = cfg[vgg_name]
        if self.rf_level == 3:
            self.maxpool = nn.MaxPool2d(kernel_size=4, stride=3, padding=1)
            self.config = cfg[vgg_name]
        if self.rf_level == 4:
            self.maxpool = nn.MaxPool2d(kernel_size=5, stride=4, padding=1)
            self.config = cfg[vgg_name]
        if self.rf_level == 5:
            self.maxpool = nn.MaxPool2d(kernel_size=6, stride=5, padding=1)
            self.config = cfg[vgg_name]
            # self.maxpool = nn.MaxPool2d(kernel_size=15, stride=14, padding=1)
        if self.rf_level == 6:
            self.maxpool = nn.MaxPool2d(kernel_size=7, stride=6, padding=1)
            self.config = cfg[vgg_name]
            # self.maxpool = nn.MaxPool2d(kernel_size=20, stride=19, padding=1)
        if self.rf_level == 7:
            self.maxpool = nn.MaxPool2d(kernel_size=8, stride=7, padding=1)
            self.config = cfg[vgg_name]
            # self.maxpool = nn.MaxPool2d(kernel_size=32, stride=31, padding=1)
        if self.rf_level == 8:
            self.maxpool = nn.MaxPool2d(kernel_size=9, stride=8, padding=1)
            self.config = cfg[vgg_name]
            # self.maxpool = nn.MaxPool2d(kernel_size=45, stride=44, padding=1)
        if self.rf_level == 9:
            self.maxpool = nn.MaxPool2d(kernel_size=10, stride=9, padding=1)
            self.config = cfg[vgg_name]
            # self.maxpool = nn.MaxPool2d(kernel_size=55, stride=54, padding=1)
        if self.rf_level == 10:
            self.maxpool = nn.MaxPool2d(kernel_size=11, stride=10, padding=1)
            self.config = cfg[vgg_name]
            # self.maxpool = nn.MaxPool2d(kernel_size=64, stride=63, padding=1)

        self.features = self._make_layers(self.config)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):

        # for i, layer in enumerate(self.features):
        #     x = layer(x)
        #     try:
        #         print("{}".format(cfg["VGG19"][i]))
        #         print("output shape of layer {}/{}: {}".format(i, len(self.features), x.size()))
        #     except:
        #
        #         print("output shape of layer {}/{}: {}".format(i, len(self.features), x.size()))
        out = self.features(x)
        # out = x
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        if self.maxpool:
            layers.insert(1, self.maxpool)
        return nn.Sequential(*layers)


def small_ResNet_rf(num_classes=10, fix_points=None, RF_level=1, multiplier=1):
    if RF_level == 0:
        return small_ResNetRF(small_Bottleneck, [1, 1, 1, 1], num_classes, fix_points)
    else:
        return small_ResNetRF(small_Bottleneck, [1, 1, 1, 1], num_classes=num_classes, fixed_points=fix_points,
                              RF_level=RF_level,
                              multiplier=multiplier)


def deep_small_ResNet_rf(num_classes=10, fix_points=None, RF_level=1, multiplier=1):
    return Deep_small_ResNetRF(small_deep_Bottleneck, [2, 2, 2, 2], num_classes=num_classes, fixed_points=fix_points,
                               RF_level=RF_level,
                               multiplier=multiplier)


def deep_2_small_Resnet_rf(num_classes=10, fix_points=None, RF_level=1, multiplier=1, number_layers=25):
    if number_layers == 20:
        return Deep_small_ResNetRF(small_deep_Bottleneck, [1, 1, 1, 1], num_classes=num_classes,
                                   fixed_points=fix_points,
                                   RF_level=RF_level,
                                   multiplier=multiplier)

    if number_layers == 25:
        return Deep_small_ResNetRF(small_deep_Bottleneck, [1, 1, 2, 1], num_classes=num_classes,
                                   fixed_points=fix_points,
                                   RF_level=RF_level,
                                   multiplier=multiplier)
    if number_layers == 40:
        return Deep_small_ResNetRF(small_deep_Bottleneck, [2, 2, 2, 2], num_classes=num_classes,
                                   fixed_points=fix_points,
                                   RF_level=RF_level,
                                   multiplier=multiplier)
    else:
        raise NotImplementedError(f"There is no model implemented for model with {number_layers} layers")


def small_ResNet_BasicBlock_rf(num_classes=10, fix_points=None, RF_level=1, multiplier=1):
    if RF_level == 0:
        return small_ResNetRF(small_BasicBlock, [1, 1, 1, 1], num_classes, fix_points)
    else:
        return small_ResNetRF(small_BasicBlock, [1, 1, 1, 1], num_classes=num_classes, fixed_points=fix_points,
                              RF_level=RF_level,
                              multiplier=multiplier)


def get_output_until_block_small_vgg(net, block, net_type=1):
    if net_type == 0:
        def features_only(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            if block == 0: return x

            x = self.layer1(x)
            if block == 1: return x

            x = self.layer2(x)
            if block == 2: return x

            # x = self.layer3(x)
            out = self.layer3[0].conv1(x)
            out = self.layer3[0].bn1(out)
            out = self.layer3[0].relu(out)

            out = self.layer3[0].conv2(out)
            out = self.layer3[0].bn2(out)
            out = self.layer3[0].relu(out)

            out = self.layer3[0].conv3(out)
            out = self.layer3[0].bn3(out)

            identity = self.layer3[0].downsample(x)

            out += identity
            out = self.layer3[0].relu(out)
            x = out
            out = self.layer3[1].conv1(x)
            # out = self.layer3[1].conv2(out)
            # def forward(self, x: Tensor) -> Tensor:
            #     identity = x
            #
            #     out = self.conv1(x)
            #     out = self.bn1(out)
            #     out = self.relu(out)
            #
            #     out = self.conv2(out)
            #     out = self.bn2(out)
            #     out = self.relu(out)
            #
            #     out = self.conv3(out)
            #     out = self.bn3(out)
            #
            #     if self.downsample is not None:
            #         identity = self.downsample(x)
            #
            #     out += identity
            #     out = self.relu(out)
            #
            #     return out
            if block == 3: return out

            x = self.layer4(x)

            return x
    else:
        def features_only(self, x):
            x = self.features(x)
            return x

    net.forward = features_only.__get__(net)  # bind method


def get_output_until_block_small_resnet(net, block, net_type=1):
    def features_only(self, x):
        # x = self.features(x)
        out = self.relu(self.bn1(self.conv1(x)))
        # if self.rf_level != 1:
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

    net.forward = features_only.__get__(net)  # bind method


def get_output_until_block_deep_small_resnet(net, block, net_type=1):
    def features_only(self, x):
        # x = self.features(x)
        out = self.relu(self.bn1(self.conv1(x)))
        # if self.rf_level != 1:
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    net.forward = features_only.__get__(net)  # bind method


def test_deep_RF_models():
    from easy_receptive_fields_pytorch.receptivefield import receptivefield, give_effective_receptive_field
    from sparse_ensemble_utils import count_parameters
    # blocks = [3, 4, 5, 6, 7, 8, 9, 10]
    blocks = [11, 12, 13]
    resnet_net = deep_small_ResNet_rf(10, RF_level=4)

    # resnet_net = deep_2_small_Resnet_rf(10, RF_level=1, number_layers=25)
    # print("parameter count resnet: {}".format(count_parameters(resnet_net)))
    # vgg_net = DeepSmallVGG_RF("deep_small_vgg_2", 10, RF_level=1)
    # print("parameter count vgg: {}".format(count_parameters(vgg_net)))
    # return

    for i in blocks:
        # samples = []
        # for j in range(10):
        # x = torch.randn(2, 3, 32, 32)
        # dummy_y = torch.zeros(3, 10)
        # dummy_y[:, 5] = 1
        # dummy_loss = nn.CrossEntropyLoss()
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
        num_layers = 25
        # print("Deep resnet level {}".format(i))
        # try:
        resnet_net = deep_2_small_Resnet_rf(10, RF_level=i, number_layers=num_layers)
        #     resnet_net(x)
        #     print(f"All good resnet with {num_layers}")
        #     vgg_net = DeepSmallVGG_RF("deep_small_vgg", 10, RF_level=i)
        #     vgg_net(x)
        #     print(f"All good resnet with {num_layers}")
        #
        # except Exception as e:
        #     print(traceback.format_exc())
        #     print(e)

        # y_resnet = resnet_net(x)
        # input_names = ['Image']
        #
        # output_names = ['y_hat']
        #
        # torch.onnx.export(resnet_net, x,
        #                   f'onnx_model_small_resnet_small_imagenet.onnx',
        #                   input_names=input_names,
        #
        # loss = dummy_loss(y_resnet, dummy_y)
        # loss.backward()
        # print(y_resnet)

        get_output_until_block_deep_small_resnet(resnet_net, block=4, net_type=1)
        # get_output_until_block_small_vgg(vgg_net, block=4, net_type=1)
        if i <= 4:
            image_size = (1, 3, 224, 224)
        else:
            image_size = (1, 3, 2200, 2200)

        resnet_rf = receptivefield(resnet_net, image_size)
        print("Receptive field of deep small ResNet Level {}".format(i))
        print(resnet_rf.rfsize)
        # vgg_rf = receptivefield(vgg_net, image_size)
        # print("Receptive field of deep small vgg Level {}".format(i))
        # print(vgg_rf.rfsize)


def models_info():
    from sparse_ensemble_utils import count_parameters
    resnet_small = small_ResNet_rf(10)
    vgg_net = small_VGG_RF('small_vgg')
    print("Small ResNet parameter count : {}".format(count_parameters(resnet_small)))
    print("Small Vgg parameter count : {}".format(count_parameters(vgg_net)))


if __name__ == '__main__':
    test_deep_RF_models()
    # models_info()