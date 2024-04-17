import argparse
from CKA_similarity.CKA import CudaCKA, CKA
import torch
import numpy as np
from torchvision import transforms as trs

from main import get_datasets, get_model


if torch.cuda.is_available():
    use_device = "cuda"
else:
    use_device = "cpu"


def main(args):
    if use_device == "cuda":
        kernel = CudaCKA("cuda")
        # similarity_matrix = torch.zeros((number_layers, number_layers), device=use_device)

    if use_device == "cpu":
        # similarity_matrix = np.zeros((number_layers, number_layers))
        kernel = CKA()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Similarity pruning experiments')
    parser.add_argument('-arch', '--architecture', type=str, default="resnet18", help='Architecture for analysis',
                        required=True)
    parser.add_argument('-dt', '--dataset', type=str, default="CIFA10", help='Dataset for analysis',
                        required=True)
    parser.add_argument('-s', '--solution', type=str, default="", help='',
                        required=False)

    parser.add_argument('-sn1', '--seedname1', type=str, default="", help='',
                        required=True)
    parser.add_argument('-sn2', '--seedname2', type=str, default="", help='',
                        required=False)
    parser.add_argument('-rfl', '--rf_level', type=int, default=1, help='',
                        required=False)
    parser.add_argument('-li', '--layer_index', type=int, default=0, help='',
                        required=False)
    parser.add_argument('-e', '--experiment', type=int, default=1, help='',
                        required=False)
    parser.add_argument('-mt1', '--modeltype1', type=str, default="alternative", help='',
                        required=False)
    parser.add_argument('-mt2', '--modeltype2', type=str, default="alternative", help='',
                        required=False)
    parser.add_argument('-ft1', '--filetype1', type=str, default="npy", help='',
                        required=False)
    parser.add_argument('-ft2', '--filetype2', type=str, default="npy", help='',
                        required=False)
    parser.add_argument('-t', '--train', type=int, default=1, help='',
                        required=False)
