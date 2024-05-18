import argparse
import math
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
# import clip
import wandb
import typing
import os
from torchvision import transforms
# from dataset import ImageNet
# from torchvision.models import resnet18, ResNet18_Weights, \
#     resnet34, ResNet34_Weights, \
#     resnet50, ResNet50_Weights, \
#     resnet152, ResNet152_Weights, \
#     mobilenet_v3_large, MobileNet_V3_Large_Weights, vit_b_32, ViT_B_32_Weights, \
#     efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet34, mobilenet_v3_large, efficientnet_b0
# import vits
import timm

print("Imported everything")


def is_prunable_module(m: torch.nn.Module):
    return (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d))


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


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_zero_parameters(model):
    return sum((p == 0).sum() for p in model.parameters() if p.requires_grad)


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


def create_feature_extractor(model):
    el_children = list(model.children())
    feature_extractor = torch.nn.Sequential(*el_children[:-2])
    return feature_extractor


def check_if_has_encode_image(model):
    invert_op = getattr(model, "invert_op", None)
    if callable(invert_op):
        invert_op(model.path.parent_op)


def prune_function(net, cfg, pr_per_layer=None):
    target_sparsity = cfg.amount
    if cfg.pruner == "global":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    if cfg.pruner == "manual":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner="manual", pr_per_layer=pr_per_layer)
        individual_prs_per_layer = prune_with_rate(net, target_sparsity,
                                                   exclude_layers=cfg.exclude_layers, type="layer-wise",
                                                   pruner="lamp", return_pr_per_layer=True)
        if cfg.use_wandb:
            log_dict = {}
            for name, elem in individual_prs_per_layer.items():
                log_dict["individual_{}_pr".format(name)] = elem
            wandb.log(log_dict)
    if cfg.pruner == "lamp":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="layer-wise",
                        pruner=cfg.pruner)


def weights_to_prune(model: torch.nn.Module, exclude_layer_list=[]):
    modules = []
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layer_list:
            modules.append((m, "weight"))
            print(name)

    return modules


def prune_with_rate(net: torch.nn.Module, amount: typing.Union[int, float], pruner: str = "erk",
                    type: str = "global",
                    criterion:
                    str =
                    "l1", exclude_layers: list = [], pr_per_layer: dict = {}, return_pr_per_layer: bool = False,
                    is_stochastic: bool = False, noise_type: str = "", noise_amplitude=0):
    if type == "global":
        print("Exclude layers in prun_with_rate:{}".format(exclude_layers))
        weights = weights_to_prune(net, exclude_layer_list=exclude_layers)
        print("Length of weigths to prune:{}".format(len(weights))
              )
        if criterion == "l1":
            prune.global_unstructured(
                weights,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        if criterion == "l2":
            prune.global_unstructured(
                weights,
                pruning_method=prune.LnStructured,
                amount=amount,
                n=2
            )
    elif type == "layer-wise":
        from layer_adaptive_sparsity.tools.pruners import weight_pruner_loader
        if pruner == "lamp":
            pruner = weight_pruner_loader(pruner)
            if return_pr_per_layer:
                return pruner(model=net, amount=amount, exclude_layers=exclude_layers,
                              return_amounts=return_pr_per_layer)
            else:
                pruner(model=net, amount=amount, exclude_layers=exclude_layers, is_stochastic=is_stochastic,
                       noise_type=noise_type, noise_amplitude=noise_amplitude)
        if pruner == "erk":
            pruner = weight_pruner_loader(pruner)
            pruner(model=net, amount=amount, exclude_layers=exclude_layers)
            # _, amount_per_layer, _, _ = erdos_renyi_per_layer_pruning_rate(model=net, cfg=cfg)
            # names, weights = zip(*get_layer_dict(net))
            # for name, module in net.named_modules():
            #     if name in exclude_layers or name not in names:
            #         continue
            #     else:
            #         prune.l1_unstructured(module, name="weight", amount=float(amount_per_layer[name]))
        if pruner == "manual":
            for name, module in net.named_modules():
                with torch.no_grad():
                    if name in exclude_layers or not is_prunable_module(module):
                        continue
                    else:
                        prune.l1_unstructured(module, name="weight", amount=float(pr_per_layer[name]))
    elif type == "random":
        weights = weights_to_prune(net, exclude_layer_list=exclude_layers)
        if criterion == "l1":
            prune.random_structured(
                weights,
                # pruning_method=prune.L1Unstructured,
                amount=amount
            )


    else:
        raise NotImplementedError("Not implemented for type {}".format(type))


if __name__ == '__main__':
    args = parse_arguments()
    from easy_receptive_fields_pytorch.receptivefield import receptivefield, give_effective_receptive_field
    from torch_receptive_field import receptive_field, receptive_field_for_unit

    # size = [1, 3, 6000, 6000]
    H, W = 4000, 4000
    size = (1 ,3, H, W)
    print(args)
    diversity_models = []
    acc_gap_models = []
    auroc_models = []
    #
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # preprocess = transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
    #
    # dataset = ImageNet(preprocess, args.data_location, args.batch_size, args.workers)

    # resnet18
    # print("ResNet18")
    # f_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # f_model.cuda()
    # f_model.eval()

    # resnet34
    print("##############################")
    print("ResNet34")
    print("##############################")
    # f_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    f_model = resnet34()
    # f_model.cuda()
    f_model.eval()
    print("Number_of_parameters:{}".format(count_parameters(f_model)))
    extractor = create_feature_extractor(f_model)
    extractor.cpu()
    le_rf = receptivefield(extractor, size)
    # le_rf = receptive_field(extractor, (3, H, W))
    # receptive_field_for_unit(le_rf, "2", (1, 1))
    print("Receptive field:\n{}".format(le_rf))

    # # # resnet152
    # weights = ResNet152_Weights.IMAGENET1K_V1
    # model_2 = resnet152(weights=weights)
    # model_2.eval()
    # #
    # # legacy_seresnet50.in1k
    # print("legacy_seresnet50.in1k")
    # s_model = timm.create_model('legacy_seresnet50.in1k', pretrained=True)
    # s_model.cuda()
    # s_model.eval()

    # legacy_seresnet34.in1k
    # print("legacy_seresnet34.in1k")
    # s_model = timm.create_model('legacy_seresnet34.in1k', pretrained=False)
    # # s_model.cuda()
    # s_model.eval()
    #
    # print("Number_of_parameters:{}".format(count_parameters(s_model)))
    # extractor = create_feature_extractor(s_model)
    # extractor.cpu()
    # le_rf = receptivefield(extractor, size)
    # print("Receptive field:\n{}".format(le_rf))

    # # resnet50
    # print("ResNet50")
    # s_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # s_model.cuda()
    # s_model.eval()

    # SK-ResNet-34

    # print("SK-ResNet-34\n")
    #
    # s_model = timm.create_model('skresnet34', pretrained=False)
    # # s_model.cuda()
    # s_model.eval()
    #
    # print("Number_of_parameters:{}".format(count_parameters(s_model)))
    # extractor = create_feature_extractor(s_model)
    # extractor.cpu()
    # le_rf = receptivefield(extractor, size)
    # print("Receptive field:\n{}".format(le_rf))

    # mobilenet-v2
    print("##############################")
    print("mobilenet-v2")
    print("##############################")
    s_model = timm.create_model('mobilenetv2_120d', pretrained=False)
    # s_model.cuda()
    s_model.eval()
    print("Number_of_parameters:{}".format(count_parameters(s_model)))
    extractor = create_feature_extractor(s_model)
    extractor.cpu()
    le_rf = receptivefield(extractor, size)
    le_rf = receptive_field(extractor, size)
    print("Receptive field:\n{}".format(le_rf))

    # le_rf = receptive_field(extractor, size)
    # print("Receptive field:\n{}".format(le_rf))
    # receptive_field_for_unit(le_rf, "2", (1, 1))
    # mobilenet-v3
    print("##############################")
    print("mobilenet-v3")
    print("##############################")
    # s_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).to("cpu")
    s_model = mobilenet_v3_large().to("cpu")
    # s_model.cuda()
    s_model.eval()

    print("Number_of_parameters:{}".format(count_parameters(s_model)))
    extractor = create_feature_extractor(s_model)
    extractor.cpu()
    le_rf = receptivefield(extractor, size)
    # le_rf = receptive_field(extractor, size)
    # print("Receptive field:\n{}".format(le_rf))
    # le_rf = receptive_field(extractor, size)
    print("Receptive field:\n{}".format(le_rf))
    # receptive_field_for_unit(le_rf, "2", (1, 1))

    # densenet
    # print("densenet")
    # s_model = timm.create_model('densenet121', pretrained=True)
    # s_model.cuda()
    # s_model.eval()

    # mnasnet_100 74.65
    # print("mnasnet")
    # s_model = timm.create_model('mnasnet_100.rmsp_in1k', pretrained=True)
    # s_model.cuda()
    # s_model.eval()
    #
    # # dpn-68
    # print("dpn")
    # s_model = timm.create_model('dpn68.mx_in1k', pretrained=True)
    # s_model.cuda()
    # s_model.eval()

    # efficientnet-b0
    print("##############################")
    print("efficientnet-b0")
    print("##############################")
    # s_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    s_model = efficientnet_b0()
    s_model.cuda()
    s_model.eval()
    extractor = create_feature_extractor(s_model)
    extractor.cpu()
    le_rf = receptivefield(extractor, size)
    print("Receptive field:\n{}".format(le_rf))

    # # vit-b/32
    # print("vit-b/32")
    # s_model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    # s_model.cuda()
    # s_model.eval()
    #
    # # mocov3 resesnet
    # print("MoCoV3 ResNet50")
    # s_model = resnet50()
    # s_model = torch.nn.DataParallel(s_model)
    # checkpoint = torch.load("linear-1000ep.pth.tar", map_location="cpu")
    # s_model.load_state_dict(checkpoint['state_dict'])
    # s_model.cuda()
    # s_model.eval()
    #
    # # mocov3 vit
    # # print("MoCoV3 ViT")
    # # s_model = vits.vit_base()
    # # s_model = torch.nn.DataParallel(s_model)
    # # checkpoint = torch.load("linear-vit-b-300ep.pth.tar", map_location="cpu")
    # # s_model.load_state_dict(checkpoint['state_dict'])
    # # s_model.cuda()
    # # s_model.eval()
    #
    #
    # # deit tiny
    # print("deit tiny distilled")
    # s_model = timm.create_model('deit_tiny_distilled_patch16_224.fb_in1k', pretrained=True)
    # s_model.cuda()
    # s_model.eval()
    #
    # # vit models
    # # vit_small_patch32_224.augreg_in21k_ft_in1k
    # base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)
    #
    # # dataset = ImageNet(preprocess, args.data_location, args.batch_size, args.workers)
    #
    #
    # print('vit_base_patch32_224.augreg_in21k_ft_in1k')
    # s_model = timm.create_model('vit_base_patch32_224.augreg_in21k_ft_in1k', pretrained=True)
    # s_model.cuda()
    # s_model = s_model.eval()
    #
    # # dataset = ImageNet(preprocess, args.data_location, args.batch_size, args.workers)
    # print("clip")
    # state_dict = torch.load('/home/chengr_lab/cse12150072/models/clip/model_0.pt', map_location=torch.device('cpu'))
    # s_model = get_model_from_sd(state_dict, base_model)
    # s_model.cuda()
    # s_model.eval()
