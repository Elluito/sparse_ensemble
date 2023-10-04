print("First line")
import time
t0 = time.time()

import os
print("After os")
# import accelerate
print("After accelerate")
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pickle
print("After Pickle")
# import paretoset
print("After pareto set")
import glob
print("After glob")
# import pygmo
import typing
print("After typing")
import pandas as pd
print("Until: 16")
import datetime as date
print("after datetime")
# import umap
print("after umap")
# import wandb
print("wandb")
#import optuna
print("After optuna")
# sys.path.append('csgmcmc')
from alternate_models import ResNet, VGG
# from csgmcmc.models import *
print("Until 20")
import omegaconf
import copy
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
print("After torchvision")
import torchvision.transforms as transforms
import scipy
import argparse
import scipy.optimize as optimize
from torch.autograd import Variable
print("After  autograd")
import numpy as np
print("After numpy")
import random
import torch.nn.utils.prune as prune
import platform
from functools import partial
print("Unitl line 40")
import glob
from torchmetrics import Accuracy
import array as pyarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FormatStrFormatter
import sklearn as sk
from sklearn.manifold import MDS, TSNE
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader, random_split, Dataset
import logging
import torchvision as tv
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.axes import SubplotBase
from itertools import chain, combinations
import seaborn as sns
import seaborn.objects as so
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from plot_utils import plot_ridge_plot, plot_double_barplot, plot_histograms_per_group, stacked_barplot, \
    stacked_barplot_with_third_subplot, plot_double_barplot
from sparse_ensemble_utils import erdos_renyi_per_layer_pruning_rate
from sparse_ensemble_utils import erdos_renyi_per_layer_pruning_rate, get_layer_dict, is_prunable_module, \
    count_parameters, sparsity, get_percentile_per_layer, get_sampler, test, restricted_fine_tune_measure_flops, \
    get_random_batch, efficient_population_evaluation, get_random_image_label, check_for_layers_collapse, get_mask, \
    apply_mask, restricted_IMAGENET_fine_tune_ACCELERATOR_measure_flops, test_with_accelerator
from itertools import cycle
from matplotlib.patches import PathPatch
import matplotlib.transforms as mtransforms
# import pylustrator
from shrinkbench.metrics.flops import flops
from pathlib import Path
import argparse
from decimal import Decimal
print("safe All imports")
t1 = time.time()
print("time for imports = {}".format(t1-t0))

