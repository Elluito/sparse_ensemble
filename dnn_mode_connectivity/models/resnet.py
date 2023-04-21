'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import typing
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import  curves

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,fix_points=None):
        super(BasicBlock, self).__init__()
        fixed_points = fix_points
        self.stride= stride
        self.planes = planes
        self.in_planes =in_planes
        if fixed_points is None:
            self.is_curve = False
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
        if fixed_points is not None:
            self.is_curve = True

            self.conv1 = curves.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,fix_points=fixed_points)
            self.bn1 = curves.BatchNorm2d(planes,fix_points=fixed_points)
            self.conv2 = curves.Conv2d(planes, planes, kernel_size=3,
                                       stride=1, padding=1, bias=False,fix_points=fixed_points)
            self.bn2 = curves.BatchNorm2d(planes,fix_points=fixed_points)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.ModuleList()
                self.shortcut.append(curves.Conv2d(in_planes, self.expansion*planes,
                                  kernel_size=1, stride=stride, bias=False,fix_points=fixed_points))
                self.shortcut.append(curves.BatchNorm2d(self.expansion*planes,fix_points=fixed_points))

    def forward_layers(self,layer,x,coef_tt):
        out = None
        assert len(layer)!=0,"There is nothing in the layer"
        for l in layer:
            if out is None:
                out = l(x, coef_tt)
            else:
                out = l(out, coef_tt)
        return out
    def forward(self, x,coef_tt=None):
        if self.is_curve:
            out = F.relu(self.bn1(self.conv1(x,coef_tt),coef_tt))
            out = self.bn2(self.conv2(out,coef_tt),coef_tt)
            if  self.stride == 1 and self.in_planes == self.expansion * self.planes:
                out += x
            else:
                out += self.forward_layers(self.shortcut,x,coef_tt)
            out = F.relu(out)

        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,fix_points=None):
        super(Bottleneck, self).__init__()

        if fix_points is None:
            self.is_curve = False
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion *
                                   planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
        if fix_points is not None:
            self.is_curve = True
            self.conv1 = curves.Conv2d(in_planes, planes, kernel_size=1, bias=False,fix_points=fix_points)
            self.bn1 = curves.BatchNorm2d(planes,fix_points=fix_points)
            self.conv2 = curves.Conv2d(planes, planes, kernel_size=3,
                                       stride=stride, padding=1, bias=False,fix_points=fix_points)
            self.bn2 = curves.BatchNorm2d(planes,fix_points=fix_points)
            self.conv3 = curves.Conv2d(planes, self.expansion *
                                       planes, kernel_size=1, bias=False,fix_points=fix_points)
            self.bn3 = curves.BatchNorm2d(self.expansion*planes,fix_points=fix_points)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.ModuleList()
                self.shortcut.append(
                    curves.Conv2d(in_planes, self.expansion*planes,
                                  kernel_size=1, stride=stride, bias=False,fix_points=fix_points))
                self.shortcut.append(
                    curves.BatchNorm2d(self.expansion*planes,fix_points=fix_points))

    def forward_layers(self,layer,x,coef_tt):
        out = None
        for l in layer:
            if out is None:
                out = l(x, coef_tt)
            else:
                out = l(out, coef_tt)
        return out

    def forward(self,inputx):
        if self.is_curve:
            x,coeffs_t = inputx
            out = F.relu(self.bn1(self.conv1(x,coeffs_t),coeffs_t))
            out = F.relu(self.bn2(self.conv2(out,coeffs_t),coeffs_t))
            out = self.bn3(self.conv3(out,coeffs_t),coeffs_t)
            out += self.forward_layers(self.shortcut,x , coeffs_t)
            out = F.relu(out)
        else:
            x = inputx
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
        return out


class ResNetBase(nn.Module):
    def __init__(self, block:typing.Union[BasicBlock,Bottleneck], num_blocks, num_classes=10):
        super(ResNetBase, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 # I did this instead of the nn.Sequential because
    def forward_call_layers(self,layer,x):
        out = None
        for l in layer :
            if out is None:
                out = l(x)
            else:
                out = l(out)
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.forward_call_layers(self.layer1,out)
        out =  self.forward_call_layers(self.layer2,out)
        out = self.forward_call_layers(self.layer3,out)
        out =  self.forward_call_layers(self.layer4,out)
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetCurve(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,fix_points=None):
        super(ResNetCurve, self).__init__()
        self.in_planes = 64
        self.fix_points = fix_points
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
        layers = nn.ModuleList()
        for stride in strides:

            layers.append(block(self.in_planes, planes, stride, fix_points = self.fix_points))
            self.in_planes = planes * block.expansion
        return layers
    def forward_call_layers(self,layer,x,coef_tt):
        out = None
        for l in layer :
            if out is None:
                out = l(x,coef_tt)
            else:
                out = l(out,coef_tt)
        return out

    def forward(self, x, coeffs_t):
        out = F.relu(self.bn1(self.conv1(x,coeffs_t),coeffs_t))
        out = self.forward_call_layers(self.layer1,out,coeffs_t)
        out =  self.forward_call_layers(self.layer2,out,coeffs_t)
        out = self.forward_call_layers(self.layer3,out,coeffs_t)
        out =  self.forward_call_layers(self.layer4,out,coeffs_t)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out,coeffs_t)
        return out

# def ResNet18(num_classes=10,fix_points=None):
#     return ResNetBase(BasicBlock, [2, 2, 2, 2],num_classes)
#
#
# def ResNet34():
#     return ResNetBase(BasicBlock, [3, 4, 6, 3])
#
#
# def ResNet50(num_classes=10,fix_points=None):
#     return ResNetBase(Bottleneck, [3, 4, 6, 3],num_classes,fix_points)
#
#
# def ResNet101():
#     return ResNetBase(Bottleneck, [3, 4, 23, 3])
#
#
# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])

class ResNet18:
    base = partial(ResNetBase,BasicBlock,[2,2,2,2])
    curve = partial(ResNetCurve,BasicBlock,[2,2,2,2])
    kwargs = {
    }
class ResNet50:
    base = partial(ResNetBase,Bottleneck, [3, 4, 6, 3])
    curve = partial(ResNetCurve,Bottleneck, [3, 4, 6, 3])
    kwargs = {
    }
# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()
