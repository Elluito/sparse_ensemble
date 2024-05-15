import argparse
import math

import torch
import clip
import os

from torchvision import transforms

from dataset import ImageNet

from torchvision.models import resnet18, ResNet18_Weights, \
    resnet34, ResNet34_Weights,\
    resnet50, ResNet50_Weights, \
    resnet152, ResNet152_Weights, \
    mobilenet_v3_large, MobileNet_V3_Large_Weights, vit_b_32, ViT_B_32_Weights, \
    efficientnet_b0, EfficientNet_B0_Weights
# import vits
import timm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('/datasets'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
    )
    return parser.parse_args()


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if initial_weights is None:
            initial_weights = torch.zeros_like(self.classification_head.weight)
            torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
        self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
        self.classification_head.bias = torch.nn.Parameter(
            torch.zeros_like(self.classification_head.bias))

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, return_features=False):
        features = self.model.encode_image(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits


def get_model_from_sd(state_dict, base_model):
    feature_dim = state_dict['classification_head.weight'].shape[1]
    num_classes = state_dict['classification_head.weight'].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    for p in model.parameters():
        p.data = p.data.float()
    model.load_state_dict(state_dict)
    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    return torch.nn.DataParallel(model, device_ids=devices)


if __name__ == '__main__':
    args = parse_arguments()
    from easy_receptive_fields_pytorch.receptivefield import receptivefield, give_effective_receptive_field
    print(args)
    diversity_models = []
    acc_gap_models = []
    auroc_models = []

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    dataset = ImageNet(preprocess, args.data_location, args.batch_size, args.workers)

    # resnet18
    print("ResNet18")
    f_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    f_model.cuda()
    f_model.eval()

    # resnet34
    print("ResNet34")
    f_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    f_model.cuda()
    f_model.eval()

    # # resnet152
    weights = ResNet152_Weights.IMAGENET1K_V1
    model_2 = resnet152(weights=weights)
    model_2.eval()
    #
    # legacy_seresnet50.in1k
    print("legacy_seresnet50.in1k")
    s_model = timm.create_model('legacy_seresnet50.in1k', pretrained=True)
    s_model.cuda()
    s_model.eval()

    # legacy_seresnet34.in1k
    print("legacy_seresnet34.in1k")
    s_model = timm.create_model('legacy_seresnet34.in1k', pretrained=True)
    s_model.cuda()
    s_model.eval()

    # resnet50
    print("ResNet50")
    s_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    s_model.cuda()
    s_model.eval()

    # SK-ResNet-34
    print("SK-ResNet-34")
    s_model = timm.create_model('skresnet34', pretrained=True)
    s_model.cuda()
    s_model.eval()

    # mobilenet-v2
    print("mobilenet-v2")
    s_model = timm.create_model('mobilenetv2_120d', pretrained=True)
    s_model.cuda()
    s_model.eval()

    # mobilenet-v3
    print("mobilenet-v3")
    s_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    s_model.cuda()
    s_model.eval()

    # densenet
    print("densenet")
    s_model = timm.create_model('densenet121', pretrained=True)
    s_model.cuda()
    s_model.eval()

    # mnasnet_100 74.65
    print("mnasnet")
    s_model = timm.create_model('mnasnet_100.rmsp_in1k', pretrained=True)
    s_model.cuda()
    s_model.eval()

    # dpn-68
    print("dpn")
    s_model = timm.create_model('dpn68.mx_in1k', pretrained=True)
    s_model.cuda()
    s_model.eval()

    # efficientnet-b0
    print("efficientnet-b0")
    s_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    s_model.cuda()
    s_model.eval()

    # vit-b/32
    print("vit-b/32")
    s_model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    s_model.cuda()
    s_model.eval()

    # mocov3 resesnet
    print("MoCoV3 ResNet50")
    s_model = resnet50()
    s_model = torch.nn.DataParallel(s_model)
    checkpoint = torch.load("linear-1000ep.pth.tar", map_location="cpu")
    s_model.load_state_dict(checkpoint['state_dict'])
    s_model.cuda()
    s_model.eval()

    # mocov3 vit
    # print("MoCoV3 ViT")
    # s_model = vits.vit_base()
    # s_model = torch.nn.DataParallel(s_model)
    # checkpoint = torch.load("linear-vit-b-300ep.pth.tar", map_location="cpu")
    # s_model.load_state_dict(checkpoint['state_dict'])
    # s_model.cuda()
    # s_model.eval()


    # deit tiny
    print("deit tiny distilled")
    s_model = timm.create_model('deit_tiny_distilled_patch16_224.fb_in1k', pretrained=True)
    s_model.cuda()
    s_model.eval()

    # vit models
    # vit_small_patch32_224.augreg_in21k_ft_in1k
    base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)
    dataset = ImageNet(preprocess, args.data_location, args.batch_size, args.workers)


    print('vit_base_patch32_224.augreg_in21k_ft_in1k')
    s_model = timm.create_model('vit_base_patch32_224.augreg_in21k_ft_in1k', pretrained=True)
    s_model.cuda()
    s_model = s_model.eval()

    # dataset = ImageNet(preprocess, args.data_location, args.batch_size, args.workers)
    print("clip")
    state_dict = torch.load('/home/chengr_lab/cse12150072/models/clip/model_0.pt', map_location=torch.device('cpu'))
    s_model = get_model_from_sd(state_dict, base_model)
    s_model.cuda()
    s_model.eval()
