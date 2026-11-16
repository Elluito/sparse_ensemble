'''DiracNet in PyTorch, with receptive-field (RF) control through a stem max-pool, for CIFAR-10/100.

Module (nn.Module) re-implementation of the functional CIFAR DiracNet in diracnets/diracnet.py, so it works with
torch.nn.utils.prune and the rest of the pruning / saturation tooling (every DiracConv2d is an nn.Conv2d whose
`weight` is the free parameter W).

Each block is  BN(conv(ReLU(x), alpha * delta + beta * normalize(W)))  with no skip connections: the Dirac delta
(identity kernel) in the weight parametrization plays the role of the residual path.

Layout, built to match ResNet50_rf stage by stage (4 stages, same spatial sizes, same number of convs per stage):
    stem conv3x3 (64) -> BN -> ReLU -> RF max-pool
    -> group0 (9 blocks, 64 ch)  -> max-pool 2 (ceil) -> group1 (12 blocks, 128 ch)
    -> max-pool 2 (ceil) -> group2 (18 blocks, 256 ch) -> max-pool 2 (ceil) -> group3 (9 blocks, 512 ch)
    -> ReLU -> global avg-pool -> fc
DiracNet-50: 9 + 12 + 18 + 9 = 48 Dirac convs (= the 3 convs x [3, 4, 6, 3] bottlenecks of ResNet-50 per stage)
+ stem + fc = 50 layers. Widths are the 3x3-conv widths of ResNet-50's bottlenecks (64 / 128 / 256 / 512).

Down-sampling uses MaxPool2d(2, ceil_mode=True), i.e. ceil(n / 2), which is what ResNet-50's stride-2 3x3 conv
(padding 1) gives: floor((n - 1) / 2) + 1. The plain DiracNet floor pooling would give 31 -> 15 instead of 16.

The RF max-pool uses the same rf_level -> (kernel_size, stride, padding) mapping as ResNetRF in resnet.py.
Feature-map sizes for a 32x32 input (group0 / group1 / group2 / group3), identical to ResNet50_rf layer1-4:
    rf_level 1: MaxPool(k=2, s=1)       31 / 16 / 8 / 4
    rf_level 2: MaxPool(k=3, s=2, p=1)  16 /  8 / 4 / 2
    rf_level 3: MaxPool(k=4, s=3, p=1)  11 /  6 / 3 / 2
    rf_level 4: MaxPool(k=5, s=4, p=1)   8 /  4 / 2 / 1

Reference:
[1] Sergey Zagoruyko, Nikos Komodakis. DiracNets: Training Very Deep Neural Networks Without Skip-Connections.
    arXiv:1706.00388
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import dirac_

# Keep `from alternate_models import *` from exporting generic names such as test() / normalize()
__all__ = ['DiracConv2d', 'DiracBlock', 'DiracNetRF', 'DiracNet50_rf', 'diracnet_parameter_groups', 'fold_dirac',
           'RF_MAXPOOL_CONFIGS']

# rf_level -> MaxPool2d(kernel_size, stride, padding), same as ResNetRF
RF_MAXPOOL_CONFIGS = {
    1: dict(kernel_size=2, stride=1, padding=0),
    2: dict(kernel_size=3, stride=2, padding=1),
    3: dict(kernel_size=4, stride=3, padding=1),
    4: dict(kernel_size=5, stride=4, padding=1),
}


def normalize(w):
    """Normalizes weight tensor over full filter."""
    return F.normalize(w.view(w.shape[0], -1)).view_as(w)


class DiracConv2d(nn.Conv2d):
    """
    Dirac parametrized convolution (diracnets/diracconv.py):  alpha * delta + beta * normalize(W),
    with learnable per-channel alpha (init 1) and beta (init 0.1) and delta the identity kernel
    (F.conv2d(x, delta) = x on the first min(in, out) channels). Only stride=1 is supported.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation,
                         bias=bias)
        self.alpha = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.full((out_channels,), 0.1))
        self.register_buffer('delta', dirac_(torch.empty_like(self.weight.data)))

    def transform_weight(self):
        return self.alpha.view(-1, 1, 1, 1) * self.delta + self.beta.view(-1, 1, 1, 1) * normalize(self.weight)

    def forward(self, input):
        return F.conv2d(input, self.transform_weight(), self.bias, self.stride, self.padding, self.dilation)


class DiracBlock(nn.Module):
    """ReLU -> DiracConv2d 3x3 -> BN (pre-activation ordering of the original DiracNet block)."""

    def __init__(self, in_planes, planes):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = DiracConv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        return self.bn(self.conv(self.relu(x)))


class DiracNetRF(nn.Module):
    # Dirac convs per stage; 50 layers: 3 convs x ResNet-50's [3, 4, 6, 3] bottlenecks, + stem + fc
    block_depths = {50: [9, 12, 18, 9]}

    def __init__(self, depth=50, width=4, num_classes=10, RF_level=1):
        super().__init__()
        if depth not in self.block_depths:
            raise ValueError(f"depth must be one of {sorted(self.block_depths)}, got {depth}")
        if RF_level not in RF_MAXPOOL_CONFIGS:
            raise ValueError(f"RF_level must be one of {sorted(RF_MAXPOOL_CONFIGS)}, got {RF_level}")
        self.rf_level = RF_level
        # width=4 -> 64 / 128 / 256 / 512, the 3x3-conv widths of ResNet-50
        self.widths = [int(v * width) for v in (16, 32, 64, 128)]

        self.conv1 = nn.Conv2d(3, self.widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.widths[0])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(**RF_MAXPOOL_CONFIGS[RF_level])

        self.in_planes = self.widths[0]
        depths = self.block_depths[depth]
        # ceil_mode=True -> ceil(n / 2), same spatial sizes as ResNet-50's stride-2 convs
        self.group0 = self._make_group(self.widths[0], depths[0])
        self.pool0 = nn.MaxPool2d(2, ceil_mode=True)
        self.group1 = self._make_group(self.widths[1], depths[1])
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)
        self.group2 = self._make_group(self.widths[2], depths[2])
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)
        self.group3 = self._make_group(self.widths[3], depths[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.widths[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.linear.bias)

    def _make_group(self, planes, count):
        layers = []
        for _ in range(count):
            layers.append(DiracBlock(self.in_planes, planes))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.group0(out)
        out = self.pool0(out)
        out = self.group1(out)
        out = self.pool1(out)
        out = self.group2(out)
        out = self.pool2(out)
        out = self.group3(out)
        out = self.avgpool(self.relu(out))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@torch.no_grad()
def fold_dirac(model):
    """
    Folds every DiracConv2d into a plain nn.Conv2d whose weight is the effective kernel
    K = alpha * delta + beta * normalize(W), in place (module names are kept, e.g. group0.3.conv).
    The network computes exactly the same function, but becomes a plain conv-BN-ReLU chain like VGG, so magnitude
    pruning ranks the kernels actually used (identity included) and pruned weights are not re-normalized.
    Call it after loading a DiracNet checkpoint and before pruning. A folded model's state_dict no longer has
    alpha / beta / delta, so it only loads into another folded model (DiracNet50_rf(...) followed by fold_dirac).
    """
    targets = [(name, m) for name, m in model.named_modules() if isinstance(m, DiracConv2d)]
    for name, m in targets:
        conv = nn.Conv2d(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride, padding=m.padding,
                         dilation=m.dilation, groups=m.groups, bias=m.bias is not None,
                         padding_mode=m.padding_mode).to(device=m.weight.device, dtype=m.weight.dtype)
        conv.weight.copy_(m.transform_weight())
        if m.bias is not None:
            conv.bias.copy_(m.bias)
        conv.train(m.training)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, conv)
    return model


def diracnet_parameter_groups(model, weight_decay=5e-4):
    """
    Optimizer parameter groups following diracnets/train.py (DiracNets-v2): weight decay only on parameters with
    more than one dimension (conv and fc weights); none on 1-D parameters (Dirac alpha / beta, BN weight / bias,
    biases). Usage: SGD(diracnet_parameter_groups(net, 5e-4), lr, momentum=0.9).
    If the weights are pruned with torch.nn.utils.prune, `weight_orig` is the 4-D parameter and still gets decay.
    """
    params_wd, params_rest = [], []
    for p in model.parameters():
        if p.requires_grad:
            (params_wd if p.dim() != 1 else params_rest).append(p)
    return [{'params': params_wd, 'weight_decay': weight_decay},
            {'params': params_rest, 'weight_decay': 0.0}]


def DiracNet50_rf(num_classes=10, rf_level=1, width=4):
    return DiracNetRF(depth=50, width=width, num_classes=num_classes, RF_level=rf_level)


def test():
    for rf_level in RF_MAXPOOL_CONFIGS:
        net = DiracNet50_rf(num_classes=10, rf_level=rf_level)
        x = torch.randn(2, 3, 32, 32)
        sizes = {}
        hooks = [getattr(net, g).register_forward_hook(lambda m, i, o, g=g: sizes.__setitem__(g, o.shape[-1]))
                 for g in ("group0", "group1", "group2", "group3")]
        y = net(x)
        for h in hooks:
            h.remove()
        n_layers = sum(isinstance(m, (nn.Conv2d, nn.Linear)) for m in net.modules())
        n_params = sum(p.numel() for p in net.parameters())
        print(f"rf_level {rf_level}: group0-3 feature maps {' / '.join(str(sizes[g]) for g in sorted(sizes))}, "
              f"output {tuple(y.shape)}, {n_layers} layers, {n_params / 1e6:.2f}M params")


if __name__ == '__main__':
    test()
