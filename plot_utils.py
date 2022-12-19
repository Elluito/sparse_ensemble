import sys
import math
import pickle
import typing
from typing import List
import pandas as pd
import datetime as date
import optuna
# sys.path.append('csgmcmc')
from csgmcmc.models import *
import omegaconf
import copy
import torchvision
import torchvision.transforms as transforms
import scipy
import os
import argparse
import scipy.optimize as optimize
from torch.autograd import Variable
import numpy as np
import random
import platform
from functools import partial
import glob
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import array as pyarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FormatStrFormatter
from sklearn.manifold import MDS, TSNE
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from torch.utils.data import DataLoader, random_split, Dataset
import logging
import torchvision as tv
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from itertools import chain, combinations
import seaborn as sns
from torch.nn.utils import parameters_to_vector, vector_to_parameters


############################### Plotting functions #####################################################################
def plot_histograms_per_group(df: pd.DataFrame, variable,group,title="",path: str="histogram.pdf"):
    fig = plt.figure()
    fig = df[variable].hist(by=df[group])
    plt.title(title,fontsize=20)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_ridge_plot(df: pd.DataFrame, path: str):
    """
    Simple wrapper for plotting a ridge plot
    :param df: DataFrame that contains the numerical value "x" and the categorical  value "g" for the ridgeplot
    :param path: path where the plot is going to be saved
    :return:
    """
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "x")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    plt.savefig(path)
    plt.close()


def plot_double_barplot(df: pd.DataFrame, ylabel1, ylabel2, title, path: str, xtick_labels: List[str], color1="blue",
                        color2="red", logy1=False, logy2=False):
    fig = plt.figure()  # Create matplotlib figure

    ax = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.
    width = 0.4
    df.y1.plot(kind='bar', color=color1, ax=ax, width=width, position=1, logy=logy1)
    df.y2.plot(kind='bar', color=color2, ax=ax2, width=width, position=0, logy=logy2)

    ax.set_ylabel(ylabel1)
    ax2.set_ylabel(ylabel2)
    ax.set_xticklabels(xtick_labels, rotation=90)
    ax2.set_xticklabels(xtick_labels, rotation=90)

    ax.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)


def fancy_bloxplot(df, x, y, hue=None, path: str = "figure.png", title="", save=True,rot=0):
    grped_bplot = sns.catplot(x=x,
                              y=y,
                              hue=hue,
                              kind="box",
                              legend=False,
                              height=6,
                              aspect=1.3,
                              data=df)  # .set(title="Title")
    # make grouped stripplot
    grped_bplot = sns.stripplot(x=x,
                                y=y,
                                hue=hue,
                                jitter=True,
                                dodge=True,
                                marker='o',
                                edgecolor="gray",
                                linewidth=1,
                                # palette="set2",
                                alpha=0.5,
                                data=df)
    # how to remove redundant legends in Python
    # Let us first get legend information from the plot object
    handles, labels = grped_bplot.get_legend_handles_labels()
    # grped_bplot.set_title("The title")
    plt.title(title, fontsize=12)
    # specify just one legend
    l = plt.legend(handles[:3], labels[:3])
    plt.xticks(rotation=rot)
    if save:
        plt.savefig(path, bbox_inches="tight")
    return plt.gcf()


def stacked_barplot(df: pd.DataFrame, x, y1, y2, ylabel, hue=None, ax=None, path: str = "figure.png", title="",
                    save=True,
                    label1="Type = one",
                    label2="Type = 2",
                    color1="darkblue",
                    color2="lightblue", rot=0, logscale=False):
    # set the figure size
    if not ax:
        plt.figure(figsize=(14, 14))
        # from raw value to percentage
        # bar chart 1 -> top bars (group of 'smoker=No')
        df["total"] = df[y1] + df[y2]
        bar1 = sns.barplot(x=x, y="total", data=df, color=color1)
        if logscale:
            bar1.set_yscale("log")

        # bar chart 2 -> bottom bars (group of 'smoker=Yes')
        bar2 = sns.barplot(x=x, y=y2, data=df, estimator=sum, errorbar=None, color=color2)
        if logscale:
            bar2.set_yscale("log")

        # add legend
        top_bar = mpatches.Patch(color=color1, label=label1)
        bottom_bar = mpatches.Patch(color=color2, label=label2)
        # leg = plt.legend(handles=[top_bar, bottom_bar])
        # add legends
        # This is for getting large patches
        leg = plt.legend(handles=[top_bar, bottom_bar], loc='upper right', labelspacing=1.5, handlelength=4)
        for patch in leg.get_patches():
            patch.set_height(22)
            patch.set_y(-6)
        plt.xticks(ticks=range(0, len(df[x])), labels=df[x], rotation=rot)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=15)
        plt.tight_layout()
        # show the graph
        if save:
            plt.savefig(path)
        else:
            return plt.gcf()
    else:
        # from raw value to percentage
        # bar chart 1 -> top bars (group of 'smoker=No')
        bar1 = sns.barplot(x=x, y=y1, data=df, ax=ax, color=color1)
        bar1.set_yscale("log")
        # bar chart 2 -> bottom bars (group of 'smoker=Yes')
        bar2 = sns.barplot(x=x, y=y2, data=df, ax=ax, color=color2)
        bar2.set_yscale("log")
        # add legend
        top_bar = mpatches.Patch(color=color1, label=label1)
        bottom_bar = mpatches.Patch(color=color2, label=label2)
        ax.legend(handles=[top_bar, bottom_bar])
        ax.set_xticklabels(rotation=rot)

        # show the graph
        if save:
            plt.savefig(path)
        else:
            return plt.gcf()


def stacked_barplot_with_third_subplot(df: pd.DataFrame, x, y1, y2, y3, label1, label2, path: str = "", title=""):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig = stacked_barplot(df, x, y1, y2, ylabel="Count", ax=axes[0], label1=label1, label2=label2, color1="blue",
                          color2="orange")
    ax2 = axes[1]
    bar3 = sns.barplot(x=x, y=y3, data=df, ax=ax2, color="darkblue")
    bar3.set_ylabel("Count")
    bar3.set_scale("log")
    ax2.suptitle(label3)
    plt.xticks(ticks=range(1, len(df[x]) + 1), labels=df[x], rotation=90)
    plt.tight_layout()
    plt.title(title, fontsize=15)
    if path:
        plt.savefig(path)
        plt.close()
    else:
        return plt.gcf()


def plot(cfg):
    performance = np.load("data/population_data/performances_{}.npy".format(cfg.noise))
    ranked_index = np.flip(np.argsort(performance))
    cutoff = 92
    plt.figure()
    plt.axhline(y=94.87, color="k", linestyle="-", label="Dense performance")
    plt.axhline(y=cutoff, color="r", linestyle="--", label="cutoff value")
    plt.xlabel("Ranking index", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    maximum = np.max(performance)
    for i, element in enumerate(performance[ranked_index]):
        if element == maximum:
            plt.scatter(i, element, c="g", marker="o", label="original model pruned")
        else:
            plt.scatter(i, element, c="b", marker="x")
    if cfg.noise == "geogaussian":
        plt.title("CIFAR10 Geometric Gaussian Noise", fontsize=20)

    if cfg.noise == "gaussian":
        plt.title("CIFAR10 Additive Gaussian Noise", fontsize=20)
    plt.legend()
    plt.show()


def example_plot(ax, fontsize=12, hide_labels=False):
    pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2.5, vmax=2.5)
    if not hide_labels:
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)
    return pc
