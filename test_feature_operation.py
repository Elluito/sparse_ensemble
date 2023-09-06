import numpy as np
import time as time
from CKA_similarity.CKA import CKA
from feature_maps_utils import load_layer_features
import torch
from main import representation_similarity_analysis

def one_product_matrix_save(prefix1, prefix2, kernel, matrix_filename,number_layers, use_cuda,index1, index2, name1="", name2=""):
    layer_i = torch.tensor(load_layer_features(prefix1, index1, name=name1))
    layer_j = torch.tensor(load_layer_features(prefix2, index2, name=name2))
    if use_cuda:
        layer_i = layer_i.cuda()
        layer_j = layer_j.cuda()

    layeri_cuda = layer_i - torch.mean(layer_i, dtype=torch.float, dim=0)
    layerj_cuda = layer_j - torch.mean(layer_j, dtype=torch.float, dim=0)


    similarity = kernel.linear_CKA(layeri_cuda.float(), layerj_cuda.float())
    np.savetxt(matrix_filename,)


if __name__ == '__main__':
    name1 = "/nobackup/sclaam/features/layer0_features_seed_1.txt"
    name2 = "/nobackup/sclaam/features/layer1_features_seed_1.txt"
    kernel = CKA()
    t0 = time.time()
    X = np.loadtxt(name1, delimiter=",")
    Y = np.loadtxt(name2, delimiter=",")
    t1 = time.time()
    print("Loading time 2 14GB files:{}".format(t1 - t0))
    print("Now the kernel operation")
    t0 = time.time()
    Similarity = kernel.linear_CKA(X, X)
    t1 = time.time()
    print("similarity: {} time lasted {}".format(Similarity, t1 - t0))
