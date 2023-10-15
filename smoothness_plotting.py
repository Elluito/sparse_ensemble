import torch
import time
import pickle
import argparse
import torchvision.transforms as transforms
from pathlib import Path
import torchvision
import torchessian
import loss_landscapes
import loss_landscapes.metrics as metrics
# libraries
import copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm

matplotlib.rcParams['figure.figsize'] = [18, 12]
# training hyperparameters
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 10 ** -2
BATCH_SIZE = 512
EPOCHS = 25
# contour plot resolution
STEPS = 40


def plot_3d(file_loss_function):
    file = open(file_loss_function, "rb")
    loss_data_fin = pickle.load(file)
    file.close()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
    Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
    ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Surface Plot of Loss Landscape')
    fig.show()
if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Smoothness plotting')
    parser.add_argument('--file', default="", type=str, help='file with loss function',required=True)
    args = vars(parser.parse_args())
    plot_3d(args["file"])