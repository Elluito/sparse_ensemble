'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import  curves
from functools import partial

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name,num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512,num_classes)

    def forward(self, x):
        out = self.features(x)
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
        return nn.Sequential(*layers)

class VGGCurve(nn.Module):
    def __init__(self, vgg_name,num_classes=10,fix_points=None):
        super(VGGCurve, self).__init__()
        self.fix_points = fix_points
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = curves.Linear(512,num_classes,fix_points=self.fix_points)

    def forward(self, x, coeffs_t):
        out = self.forward_call_layers(self.features,x,coeffs_t)
        out = out.view(out.size(0), -1)
        out = self.classifier(out,coeffs_t)
        return out

    def forward_call_layers(self,layer,x,coef_tt):
        out = None
        for l in layer :
            if out is None:
                out = l(x,coef_tt)
            else:
                out = l(out,coef_tt)
        return out

    def _make_layers(self, cfg):
        layers = nn.ModuleList()
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers +=[nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [curves.Conv2d(in_channels, x, kernel_size=3, padding=1,fix_points=self.fix_points),
                           curves.BatchNorm2d(x,fix_points=self.fix_points),

                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return layers



def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())


class VGG19:
    base = partial(VGG,"VGG19")
    curve = partial(VGGCurve,"VGG19")
    kwargs = {
        # 'depth': 19,
        # 'batch_norm': False
    }

# test()
