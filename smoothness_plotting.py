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


def main(file_loss_function, name, title):
    file = open(file_loss_function, "rb")
    loss_data_fin = pickle.load(file)
    file.close()
    plot_3d(loss_data_fin=loss_data_fin, name=name, title=title)
    countour_plot(loss_data_fin, title, name)


def countour_plot(loss_data_fin, title, name):
    plt.figure()
    # X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
    # Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
    plt.contour(loss_data_fin, levels=150)
    plt.title('Loss Contours around {}'.format(title))
    plt.savefig("smoothness_plots/countour_plot_{}.png".format(name))
    plt.close()


def plot_3d(loss_data_fin, name, title, save=True):
    # for ii in range(0, 360, 1):
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #     X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
    #     Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
    #     ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    #     ax.set_title('Surface Plot of {}'.format(name))
    #     ax.view_init(elev=10., azim=ii)
    #     print(ii)
    #     plt.savefig("movie%d.png" % ii)
    #     plt.close()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
    Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
    ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Surface Plot of {}'.format(title))
    ax.view_init(elev=10., azim=7)
    if save:
        plt.savefig("smoothness_plots/loss_landscape_{}.png".format(name))
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smoothness plotting')
    parser.add_argument('--file', default="", type=str, help='file with loss function', required=True)
    parser.add_argument('--name', default="", type=str, help='Name of the of the plot file', required=True)
    parser.add_argument('--title', default="", type=str, help='Personalised title of the plots', required=True)
    args = vars(parser.parse_args())
    main(args["file"], args["name"], args["title"])
