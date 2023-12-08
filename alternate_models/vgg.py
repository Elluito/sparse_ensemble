'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_rf': [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, 512, 512, 512, 512],
    'VGG19_rf_mod': [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

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


class VGG_RF(nn.Module):
    def __init__(self, vgg_name, num_classes=10, rf_level=0):
        super(VGG_RF, self).__init__()
        self.rf_level = rf_level
        self.maxpool = None
        self.config = None
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
            self.config = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]

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


def test():
    # net = VGG('VGG11')
    # x = torch.randn(2,3,32,32)
    # y = net(x)
    # print(y.size())

    from easy_receptive_fields_pytorch.receptivefield import receptivefield, give_effective_receptive_field
    import matplotlib.pyplot as plt
    from main import get_features_only_until_block_layer_VGG
    from feature_maps_utils import get_activations_shape
    from torchvision.models import resnet18, resnet50

    print("Receptive field normal resnet18")
    # net = ResNet18_rf(num_classes=10, rf_level=4)

    # net = resnet18(pretrained=False)

    # net.classifier = torch.nn.Linear(512, 10)

    # net.to(device)
    # x = torch.randn(3, 3, 32, 32)
    # shapes = get_activations_shape(net, x)
    # y = net(torch.randn(3, 3, 32, 32))
    # print(y)
    # blocks = [1, 2, 3, 4]
    blocks = [5]
    receptive_fields = []

    # net = VGG_RF('VGG19_rf', rf_level=4)
    # y = net(torch.randn(3, 3, 32, 32))
    # print(y)
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    for i in blocks:
        # samples = []
        # for j in range(10):
        net = VGG_RF('VGG19_rf', rf_level=i)
        x = torch.randn(3, 3, 32, 32)
        y = net(x)
        print(y)
        get_features_only_until_block_layer_VGG(net, block=4, net_type=1)
        le_rf = receptivefield(net, (1, 3, 1500, 1500))
        print(le_rf)
        # in_grad, input_pos = give_effective_receptive_field(net, (1, 3, 32, 32))

        # samples.append(torch.abs(in_grad))
    # stacked_samples = torch.stack(samples, dim=0)
    # mean = stacked_samples.mean(dim=0)
    # axs[i-1].matshow(mean, cmap="magma")
    # axs[i-1].set_title("RF {}".format(i))
    # axs[i-1].colorbar()
    # plt.vlines(x=input_pos[0] - 16, ymin=input_pos[1] - 16, ymax=input_pos[1] + 16, color='r')
    # plt.vlines(x=input_pos[0] + 16, ymin=input_pos[1] - 16, ymax=input_pos[1] + 16, color='r')
    # plt.hlines(y=input_pos[1] - 16, xmin=input_pos[0] - 16, xmax=input_pos[0] + 16, color='r')
    # plt.hlines(y=input_pos[1] + 16, xmin=input_pos[0] - 16, xmax=input_pos[0] + 16, color='r')
    # rf = receptivefield(net, (1, 3, 1000, 1000))

    # pdb.set_trace()

    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    test()
# test()
