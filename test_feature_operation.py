import numpy as np
import time as time
from CKA_similarity import CKA

if __name__ == '__main__':
    name1 = "/nobackup/sclaam/features/layer0_features_seed_1.txt"
    name2 = "/nobackup/sclaam/features/layer1_features_seed_1.txt"
    kernel = CKA()
    t0 = time.time()
    X = np.loadtxt(name1)
    Y = np.loadtxt(name2)
    t1 = time.time()
    print("Loading time 2 14GB files:{}".format(t1 - t0))
    print("Now the kernel operation")
    t0 = time.time()
    Similarity = kernel.linear_CKA(X, X)
    t1 = time.time()
    print("similarity: {}")
