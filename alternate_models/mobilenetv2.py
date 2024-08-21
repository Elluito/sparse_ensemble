'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2_cifar(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2_cifar, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
class MobileNetV2_imagenet(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10  | | change stride 1 -> 2 for ImageNet
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2_imagenet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 4 -> 7 for IMAGENET
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
class MobileNetV2_cifar_RF(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10,RF_level=0):
        super(MobileNetV2_cifar_RF, self).__init__()
        self.rf_level=RF_level
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

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
class MobileNetV2_imagenet_RF(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10  | | change stride 1 -> 2 for ImageNet
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10,RF_level=0):
        super(MobileNetV2_imagenet_RF, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10, || NOTE: change conv1 stride 1 -> 2 for IMagenet
        self.rf_level=RF_level
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 4 -> 7 for IMAGENET
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
def get_features_mobilenetv2(net):
    def features_only(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpooling(x)
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        return out

    net.forward = features_only.__get__(net)  # bind method


    net.forward = features_only.__get__(net)  # bind method
def test():
    for i in [1,2,3,4,5,6,7,8,9,10,11]:
        try:
            net = MobileNetV2_cifar_RF(10,RF_level=i)
            x = torch.randn(2,3,32,32)
            y = net(x)
            print(y.size())
        except Exception as e:
            print("RF level: {} is too big for imagesize of 32x32".format(i))
            print(e)

        try:
            net = MobileNetV2_cifar_RF(10,RF_level=i)
            x = torch.randn(2,3,64,64)
            y = net(x)
            print(y.size())
        except Exception as e:
            print("RF level: {} is too big for imagesize of 64x64".format(i))
            print(e)
# test()
if __name__ == '__main__':
    test()