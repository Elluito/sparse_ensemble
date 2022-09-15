import sys
import pickle
from typing import List
import pandas as pd
sys.path.append('csgmcmc')
from csgmcmc.models import *
import omegaconf
import copy
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import hydra
import torchvision
import torchvision.transforms as transforms
import scipy
import os
import argparse
import scipy.optimize as optimize
from models import *
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.utils.prune as prune
import platform
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import sklearn as sk
from sklearn.manifold import MDS, TSNE
from collections import defaultdict



def layer_importance_exp(model,type="a"):

 pass

