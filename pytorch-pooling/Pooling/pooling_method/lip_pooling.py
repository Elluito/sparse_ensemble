import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def lip2d(x, logit, kernel=2, stride=2, padding=0):
    weight = logit.exp()
    return F.avg_pool2d(x*weight, kernel, stride, padding)/F.avg_pool2d(weight, kernel, stride, padding)

class SoftGate(nn.Module):
    def __init__(self, coeff=12):
        super(SoftGate, self).__init__()
        self.coeff = coeff

    def forward(self, x):
        return torch.sigmoid(x).mul(self.coeff)

class SimplifiedLIP(nn.Module):
    def __init__(self, channels, kernel_size=2, stride=None, padding=0):
        super(SimplifiedLIP, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

        rp = channels
        # nn.Sequential + OrderedDict 可以为每一个层赋予一个名字，来替代掉本身的 index 命名， 但是只能通过index访问
        self.logit = nn.Sequential(
            OrderedDict((
                ('conv', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
                ('bn', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )

    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x), kernel=self.kernel_size, stride=self.stride, padding=self.padding)
        return frac

def lip():
    print("You are using Lip Pooling Method")
    return SimplifiedLIP
