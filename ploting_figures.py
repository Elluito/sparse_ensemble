import re
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
sns.reset_orig()
sns.reset_defaults()
matplotlib.rc_file_defaults()

plt.rcParams.update({
    "axes.linewidth": 0.5,
    'axes.edgecolor': 'black',
    "grid.linewidth": 0.4,
    "lines.linewidth": 1,
    'xtick.bottom': True,
    'xtick.color': 'black',
    "xtick.direction": "out",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    'ytick.left': True,
    'ytick.color': 'black',
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "figure.figsize": [3.3, 2.5],
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{bm} \usepackage{amsmath}",
})

fs = 12
import matplotlib.lines as mlines
import pandas as pd
import torch
import numpy as np
import matplotlib.patches as mpatches
import torchessian as torchessian
import pickle

def main():
    #################### Plot training #############################################

    hfont = {'fontname': 'Times New Roman'}
    # plt.rcParams["font.family"] = "Times New Roman"
    # import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.figure()
    rf2 = pd.read_csv("resnet50_normal_cifar10_200_epoch_rf_level_2_recording.csv", delimiter=",")
    rf7 = pd.read_csv("resnet50_normal_cifar10_200_epoch_rf_level_7_recording.csv", delimiter=",")
    rf7_2 = pd.read_csv("resnet50_normal_cifar10_400_epoch_rf_level_7_recording.csv", delimiter=",")
    rf7_3 = pd.read_csv("resnet50_normal_cifar10_400_epochs_rf_level_7_recording_tmax_=_epochs.csv", delimiter=",")


    def get_accuracy(el_string):
        new_string = el_string.split(" ")[0]
        list_of_string = re.findall("\d+\.\d+", el_string)
        # print(list_of_string[0])
        return float(list_of_string[0])


    # rf2.apply(get_accuracy,axis=2)
    rf2["Training Accuracy"] = rf2["training accuracy"].apply(get_accuracy)
    rf7["Training Accuracy"] = rf7["training accuracy"].apply(get_accuracy)
    rf7_2["Training Accuracy"] = rf7_2["training accuracy"].apply(get_accuracy)
    rf7_3["Training Accuracy"] = rf7_3["training accuracy"].apply(get_accuracy)

    rf2["Test Accuracy"] = rf2["test accuracy"]
    rf7["Test Accuracy"] = rf7["test accuracy"]
    rf7_2["Test Accuracy"] = rf7_2["test accuracy"]
    rf7_3["Test Accuracy"] = rf7_3["test accuracy"]

    # sns.plot(data=all_df, x="Dense Accuracy", y="Pruned Fine Tuned Accuracy", hue="Receptive Field",size="Receptive Field",sizes=(20, 200))
    fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(10, 7))

    plt.plot(rf2["Epoch"], rf2["Test Accuracy"], label="RF 213 Test")
    plt.plot(rf2["Epoch"], rf2["Training Accuracy"], label="RF 213 Training")
    # plt.plot(rf7["Epoch"],rf7["Test Accuracy"],label="RF 3100 Test")
    # plt.plot(rf7["Epoch"],rf7["Training Accuracy"],label="RF 3100 Training ")
    # plt.plot(rf7_2["Epoch"],rf7_2["Test Accuracy"],label="RF 3100 Test 2X epochs ")
    # plt.plot(rf7_2["Epoch"],rf7_2["Training Accuracy"],label="RF 3100 Training 2X epochs")
    plt.plot(rf7_3["Epoch"], rf7_3["Test Accuracy"], label="RF 3100 Test")
    plt.plot(rf7_3["Epoch"], rf7_3["Training Accuracy"], label="RF 3100 Train")
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    # plt.xlim(-5,750)

    plt.legend(prop={"size": 15},loc="lower right")
    plt.tight_layout()
    plt.savefig("paper_plots/resnet50_CIFAR10_training.pdf", bbox_inches="tight")

    ### Tiny ImageNet
    plt.figure()
    rf2 = pd.read_csv("resnet50_normal_tiny_imagenet_300_epoch_rf_level_2_recording.csv", delimiter=",")
    rf7 = pd.read_csv("resnet50_normal_tiny_imagenet_300_epoch_rf_level_7_recording.csv", delimiter=",")


    def get_accuracy(el_string):
        new_string = el_string.split(" ")[0]
        list_of_string = re.findall("\d+\.\d+", el_string)
        # print(list_of_string[0])
        return float(list_of_string[0])


    # rf2.apply(get_accuracy,axis=2)
    rf2["Training Accuracy"] = rf2["training accuracy"].apply(get_accuracy)
    rf7["Training Accuracy"] = rf7["training accuracy"].apply(get_accuracy)

    rf2["Test Accuracy"] = rf2["test accuracy"]
    rf7["Test Accuracy"] = rf7["test accuracy"]

    # sns.plot(data=all_df, x="Dense Accuracy", y="Pruned Fine Tuned Accuracy", hue="Receptive Field",size="Receptive Field",sizes=(20, 200))

    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(10, 7))

    plt.plot(rf2["Epoch"], rf2["Test Accuracy"], label="RF 213 Test")
    plt.plot(rf2["Epoch"], rf2["Training Accuracy"], label="RF 213 Training")
    plt.plot(rf7["Epoch"], rf7["Test Accuracy"], label="RF 3100 Test")
    plt.plot(rf7["Epoch"], rf7["Training Accuracy"], label="RF 3100 Training")
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    # plt.legend()
    plt.xlim(-5, 200)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    plt.legend(prop={"size": 15},loc="upper left")
    plt.tight_layout()
    plt.savefig("paper_plots/resnet50_tiny_ImageNet_training.pdf", bbox_inches="tight")
    #################### Similarity Plots #############################################

    """# ResNet50 receptive field levels in trained models

    ## Level 1 receptive field
    """

    m1 = np.loadtxt(
        "similarity_experiments/resnet50__seed_1_rf_level_1_V__seed_2_rf_level_1_.txt",
        delimiter=",")

    # similarity_experiments/resnet50__seed_1_rf_level_1_V__seed_1_rf_level_2_.txt

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(m1, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    ax.axhline(y=24.5, color='g', linewidth=2)
    ax.axvline(x=24.5, color='g', linewidth=2)
    plt.savefig("paper_plots/resnet50_level1_similarity_cifar10.pdf", bbox_inches="tight")


    """### In block similarities"""

    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    sim_inbloc_v_inblock = m1[in_block_layers, :]
    sim_inbloc_v_inblock = sim_inbloc_v_inblock[:, in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_inbloc_v_inblock, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(32), in_block_layers, rotation=90)
    plt.yticks(range(32), in_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level1_in_block_similarity_cifar10.pdf", bbox_inches="tight")


    """### Out block similarities"""

    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m1[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, out_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(out_block_layers)), out_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level1_out_block_similarity_cifar10.pdf", bbox_inches="tight")


    """###  IN and out of block similairties"""

    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m1[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, in_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(in_block_layers)), in_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level1_in_out_block_similarity_cifar10.pdf", bbox_inches="tight")


    """## Level 2 receptive field

    """

    m2 = np.loadtxt(
        "similarity_experiments/resnet50__seed_1_rf_level_2_V__seed_2_rf_level_2_.txt",
        delimiter=",")
    # similarity_experiments/resnet50__seed_1_rf_level_1_V__seed_1_rf_level_2_.txt

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(m2, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    ax.axhline(y=24.5, color='g', linewidth=2)
    ax.axvline(x=24.5, color='g', linewidth=2)
    plt.savefig("paper_plots/resnet50_level2_similarity_cifar10.pdf", bbox_inches="tight")


    """### In block similarities"""

    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    sim_inbloc_v_inblock = m2[in_block_layers, :]
    sim_inbloc_v_inblock = sim_inbloc_v_inblock[:, in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_inbloc_v_inblock, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(32), in_block_layers, rotation=90)
    plt.yticks(range(32), in_block_layers)
    plt.savefig("paper_plots/resnet50_level2_in_similarity_cifar10.pdf", bbox_inches="tight")
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)


    """### Out block similarities"""

    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m2[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, out_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(out_block_layers)), out_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level2_out_similarity_cifar10.pdf", bbox_inches="tight")


    """###  IN and aout of block similairties"""

    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m2[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, in_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(in_block_layers)), in_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level2_in_out_similarity_cifar10.pdf", bbox_inches="tight")


    """## Level 3 receptive field"""

    m3 = np.loadtxt(
        "similarity_experiments/resnet50__seed_1_rf_level_3_V__seed_2_rf_level_3_.txt",
        delimiter=",")
    # similarity_experiments/resnet50__seed_1_rf_level_1_V__seed_1_rf_level_2_.txt

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(m3, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    ax.axhline(y=24.5, color='g', linewidth=2)
    ax.axvline(x=24.5, color='g', linewidth=2)
    plt.savefig("paper_plots/resnet50_level3_similarity_cifar10.pdf", bbox_inches="tight")


    """### In block similarities"""

    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    sim_inbloc_v_inblock = m3[in_block_layers, :]
    sim_inbloc_v_inblock = sim_inbloc_v_inblock[:, in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_inbloc_v_inblock, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(32), in_block_layers, rotation=90)
    plt.yticks(range(32), in_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level3_in_similarity_cifar10.pdf", bbox_inches="tight")


    """### Out block similarities"""

    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m3[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, out_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(out_block_layers)), out_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level3_out_similarity_cifar10.pdf", bbox_inches="tight")


    """###  In and out of block similarities"""

    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m3[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, in_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(in_block_layers)), in_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level3_in_out_similarity_cifar10.pdf", bbox_inches="tight")


    """## Level 4 receptive field"""

    m4 = np.loadtxt(
        "similarity_experiments/resnet50__seed_1_rf_level_4_V__seed_2_rf_level_4_.txt",
        delimiter=",")
    # similarity_experiments/resnet50__seed_1_rf_level_1_V__seed_1_rf_level_2_.txt

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(m4, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(m4)), rotation=90)
    plt.yticks(range(len(m4)))
    ax.axhline(y=24.5, color='g', linewidth=2)
    ax.axvline(x=24.5, color='g', linewidth=2)
    plt.savefig("paper_plots/resnet50_level4_similarity_cifar10.pdf", bbox_inches="tight")


    """### In block similarities"""

    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    sim_inbloc_v_inblock = m4[in_block_layers, :]
    sim_inbloc_v_inblock = sim_inbloc_v_inblock[:, in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_inbloc_v_inblock, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(32), in_block_layers, rotation=90)
    plt.yticks(range(32), in_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)

    plt.savefig("paper_plots/resnet50_level4_in_similarity_cifar10.pdf", bbox_inches="tight")

    """### Out block similarities"""

    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m4[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, out_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(out_block_layers)), out_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level4_out_similarity_cifar10.pdf", bbox_inches="tight")


    """### IN block and out block similarities"""

    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m4[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, in_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(in_block_layers)), in_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level4_in_out_similarity_cifar10.pdf", bbox_inches="tight")


    # Mean similarity after layer 25 for all levels
    half_m1 = m1[24:, 24:]
    half_m2 = m2[24:, 24:]
    half_m3 = m3[24:, 24:]
    half_m4 = m4[24:, 24:]
    print("Upper half")
    print("Mean \"Similarity\" for half m1 {}".format(np.mean(half_m1)))
    print("Mean \"Similarity\" for half m2 {}".format(np.mean(half_m2)))
    print("Mean \"Similarity\" for half m3 {}".format(np.mean(half_m3)))
    print("Mean \"Similarity\" for half m4 {}".format(np.mean(half_m4)))
    print("Lower Half")
    half_m1 = m1[:24, 24:]
    half_m2 = m2[:24, 24:]
    half_m3 = m3[:24, 24:]
    half_m4 = m4[:24, 24:]
    print("Mean \"Similarity\" for half m1 {}".format(np.mean(half_m1)))
    print("Mean \"Similarity\" for half m2 {}".format(np.mean(half_m2)))
    print("Mean \"Similarity\" for half m3 {}".format(np.mean(half_m3)))
    print("Mean \"Similarity\" for half m4 {}".format(np.mean(half_m4)))

    """## Level 5"""

    m5_1 = np.loadtxt(
        "similarity_experiments/resnet50_trained_seed_0_rf_level_5_V_trained_seed_1_rf_level_5_.txt",
        delimiter=",")
    m5_2 = np.loadtxt(
        "similarity_experiments/resnet50_trained_seed_0_rf_level_5_V_trained_seed_2_rf_level_5_.txt",
        delimiter=",")
    m5_3 = np.loadtxt(
        "similarity_experiments/resnet50_trained_seed_1_rf_level_5_V_trained_seed_2_rf_level_5_.txt",
        delimiter=",")
    m5 = (m5_1 + m5_2 + m5_3) / 3

    # similarity_experiments/resnet50__seed_1_rf_level_1_V__seed_1_rf_level_2_.txt

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(m5, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # ax.tick_params(range(49))
    plt.xticks(range(len(m5)), rotation=90)
    plt.yticks(range(len(m5)))
    # plt.colorbar()
    ax.axhline(y=24.5, color='g', linewidth=2)
    ax.axvline(x=24.5, color='g', linewidth=2)
    plt.savefig("paper_plots/resnet50_level5_similarity_cifar10.pdf", bbox_inches="tight")


    """### In block similarities"""

    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    sim_inbloc_v_inblock = m5[in_block_layers, :]
    sim_inbloc_v_inblock = sim_inbloc_v_inblock[:, in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_inbloc_v_inblock, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.colorbar()
    plt.xticks(range(32), in_block_layers, rotation=90)
    plt.yticks(range(32), in_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level5_in_similarity_cifar10.pdf", bbox_inches="tight")


    """### Out block similarities"""

    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m5[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, out_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(out_block_layers)), out_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level5_out_similarity_cifar10.pdf", bbox_inches="tight")


    """### IN block and out block similarities"""

    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m5[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, in_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(in_block_layers)), in_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level5_in_out_similarity_cifar10.pdf", bbox_inches="tight")


    """## Level 6"""

    m6_1 = np.loadtxt(
        "similarity_experiments/resnet50_trained_seed_0_rf_level_6_V_trained_seed_1_rf_level_6_.txt",
        delimiter=",")
    m6_2 = np.loadtxt(
        "similarity_experiments/resnet50_trained_seed_0_rf_level_6_V_trained_seed_2_rf_level_6_.txt",
        delimiter=",")
    m6_3 = np.loadtxt(
        "similarity_experiments/resnet50_trained_seed_1_rf_level_6_V_trained_seed_2_rf_level_6_.txt",
        delimiter=",")
    m6 = (m6_1 + m6_2 + m6_3) / 3
    # similarity_experiments/resnet50__seed_1_rf_level_1_V__seed_1_rf_level_2_.txt

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(m6, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(m6)), rotation=90)
    plt.yticks(range(len(m6)))
    ax.axhline(y=24.5, color='g', linewidth=2)
    ax.axvline(x=24.5, color='g', linewidth=2)
    plt.savefig("paper_plots/resnet50_level6_similarity_cifar10.pdf", bbox_inches="tight")

    print("M6_1 row 2:{} with min of {:0.26f} and max of {:0.26f} and mean {:0.26f}".format(m6_1[2, :],
                                                                                            np.min(m6_1[2, :]),
                                                                                            np.max(m6_1[2, :]),
                                                                                            np.mean(m6_1[2, :])))
    print("M6_2 row 2:{} with min of {:0.26f} and max of {:0.26f} and mean {:0.26f}".format(m6_2[2, :],
                                                                                            np.min(m6_2[2, :]),
                                                                                            np.max(m6_2[2, :]),
                                                                                            np.mean(m6_2[2, :])))
    print("M6_3 row 2:{} with min of {:0.26f} and max of {:0.26f} and mean {:0.26f}".format(m6_3[2, :],
                                                                                            np.min(m6_3[2, :]),
                                                                                            np.max(m6_3[2, :]),
                                                                                            np.mean(m6_3[2, :])))

    """### In block similarities"""

    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    sim_inbloc_v_inblock = m6[in_block_layers, :]
    sim_inbloc_v_inblock = sim_inbloc_v_inblock[:, in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_inbloc_v_inblock, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(32), in_block_layers, rotation=90)
    plt.yticks(range(32), in_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level6_in_similarity_cifar10.pdf", bbox_inches="tight")


    """### Out block similarities"""

    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m6[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, out_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(out_block_layers)), out_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level6_out_similarity_cifar10.pdf", bbox_inches="tight")


    """### In block and outblock similarities

    """

    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m6[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, in_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.colorbar()
    plt.xticks(range(len(in_block_layers)), in_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level6_in_out_similarity_cifar10.pdf", bbox_inches="tight")


    """## Level 7

    """

    m7_1 = np.loadtxt(
        "similarity_experiments/resnet50_trained_seed_0_rf_level_7_V_trained_seed_1_rf_level_7_.txt",
        delimiter=",")
    m7_2 = np.loadtxt(
        "similarity_experiments/resnet50_trained_seed_0_rf_level_7_V_trained_seed_2_rf_level_7_.txt",
        delimiter=",")
    m7_3 = np.loadtxt(
        "similarity_experiments/resnet50_trained_seed_1_rf_level_7_V_trained_seed_2_rf_level_7_.txt",
        delimiter=",")
    # similarity_experiments/resnet50__seed_1_rf_level_1_V__seed_1_rf_level_2_.txt
    m7 = (m7_1 + m7_2 + m7_3) / 3

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(m7, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.colorbar()
    plt.xticks(range(len(m7)), rotation=90)
    plt.yticks(range(len(m7)))
    ax.axhline(y=24.5, color='g', linewidth=2)
    ax.axvline(x=24.5, color='g', linewidth=2)
    plt.savefig("paper_plots/resnet50_level7_similarity_cifar10.pdf", bbox_inches="tight")


    """### In block similarities"""

    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    sim_inbloc_v_inblock = m7[in_block_layers, :]
    sim_inbloc_v_inblock = sim_inbloc_v_inblock[:, in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_inbloc_v_inblock, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.colorbar()
    plt.xticks(range(32), in_block_layers, rotation=90)
    plt.yticks(range(32), in_block_layers)
    plt.savefig("paper_plots/resnet50_level7_in_similarity_cifar10.pdf", bbox_inches="tight")
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)


    """### Out block similarities"""

    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m7[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, out_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.colorbar()
    plt.xticks(range(len(out_block_layers)), out_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)
    plt.savefig("paper_plots/resnet50_level7_out_similarity_cifar10.pdf", bbox_inches="tight")


    """### IN and out block similarities"""

    in_block_layers = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38,
                       40, 41, 43, 44, 46, 47]
    out_block_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    sim_out_V_out = m7[out_block_layers, :]
    sim_out_V_out = sim_out_V_out[:, in_block_layers]
    # sim_inbloc_v_inblock=m4[in_block_layers,in_block_layers]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(sim_out_V_out, cmap='magma', vmin=0, vmax=1)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.colorbar()
    plt.xticks(range(len(in_block_layers)), in_block_layers, rotation=90)
    plt.yticks(range(len(out_block_layers)), out_block_layers)
    plt.savefig("paper_plots/resnet50_level7_in_out_similarity_cifar10.pdf", bbox_inches="tight")
    # ax.axhline(y = 24.5, color='g',linewidth = 2)
    # ax.axvline(x = 24.5, color='g',linewidth = 2)


    ############## Hessian Spectra  ####################################################
    m = 90
    ### Trained
    ############ ResNet
    l_file = open("smoothness/resnet50/l__seed_2_rf_level_0.pkl", "rb")
    w_file = open("smoothness/resnet50/w__seed_2_rf_level_0.pkl", "rb")
    l_s2_level0 = pickle.load(l_file)
    w_s2_level0 = pickle.load(w_file)
    w_s2_level0 = torch.tensor(list([w_s2_level0[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()

    l_file = open("smoothness/resnet50/l__seed_1_rf_level_1.pkl", "rb")
    w_file = open("smoothness/resnet50/w__seed_1_rf_level_1.pkl", "rb")
    l_s1_level1 = pickle.load(l_file)
    w_s1_level1 = pickle.load(w_file)
    w_s1_level1 = torch.tensor(list([w_s1_level1[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()
    l_file = open("smoothness/resnet50/l__seed_1_rf_level_2.pkl", "rb")
    w_file = open("smoothness/resnet50/w__seed_1_rf_level_2.pkl", "rb")
    l_s1_level2 = pickle.load(l_file)
    w_s1_level2 = pickle.load(w_file)
    w_s1_level2 = torch.tensor(list([w_s1_level2[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()

    l_file = open("smoothness/resnet50/l__seed_2_rf_level_2.pkl", "rb")
    w_file = open("smoothness/resnet50/w__seed_2_rf_level_2.pkl", "rb")

    l_s2_level2 = pickle.load(l_file)
    w_s2_level2 = pickle.load(w_file)
    w_s2_level2 = torch.tensor(list([w_s2_level2[0, i] ** 2 for i in range(m)]))

    w_file.close()
    l_file.close()

    l_file = open("smoothness/resnet50/l__seed_1_rf_level_3.pkl", "rb")
    w_file = open("smoothness/resnet50/w__seed_1_rf_level_3.pkl", "rb")
    l_s1_level3 = pickle.load(l_file)
    w_s1_level3 = pickle.load(w_file)
    w_s1_level3 = torch.tensor(list([w_s1_level3[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()

    l_file = open("smoothness/resnet50/l__seed_2_rf_level_3.pkl", "rb")
    w_file = open("smoothness/resnet50/w__seed_2_rf_level_3.pkl", "rb")
    l_s2_level3 = pickle.load(l_file)
    w_s2_level3 = pickle.load(w_file)
    w_s2_level3 = torch.tensor(list([w_s2_level3[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()

    l_file = open("smoothness/resnet50/l__seed_1_rf_level_4.pkl", "rb")
    w_file = open("smoothness/resnet50/w__seed_1_rf_level_4.pkl", "rb")
    l_s1_level4 = pickle.load(l_file)
    w_s1_level4 = pickle.load(w_file)
    w_s1_level4 = torch.tensor(list([w_s1_level4[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()
    l_file = open("smoothness/resnet50/l__seed_1_rf_level_p.pkl", "rb")
    w_file = open("smoothness/resnet50/w__seed_1_rf_level_p.pkl", "rb")
    l_s1_levelp = pickle.load(l_file)
    w_s1_levelp = pickle.load(w_file)
    w_s1_levelp = torch.tensor(list([w_s1_levelp[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()

    support = torch.linspace(-11, 18, 10000)
    fig, ax = plt.subplots(figsize=(15, 7))
    density = torchessian.F(support, l_s2_level0.real, w_s2_level0.real, m)
    ax.plot(support.numpy(), density.numpy(), color='m')
    # density = torchessian.F(support,l_s1_level1.real,w_s1_level1.real,m)
    # ax.plot(support.numpy(), density.numpy(), color='c')
    # density = torchessian.F(support, l_s2_level2.real,w_s2_level2.real, m)
    # ax.plot(support.numpy(), density.numpy(), color='r')
    density = torchessian.F(support, l_s1_level3.real, w_s1_level3.real, m)
    ax.plot(support.numpy(), density.numpy(), color='b')
    density = torchessian.F(support, l_s1_level4.real, w_s1_level4.real, m)
    ax.plot(support.numpy(), density.numpy(), color='g')
    # density = torchessian.F(support,l_s1_levelp.real,w_s1_levelp.real,m)
    # ax.plot(support.numpy(), density.numpy(), color='y')
    ax.set_yscale('log')
    ax.set_yticks([10 ** (i - 7) for i in range(10)])
    ax.set_ylim(10 ** -10, 10 ** 3)
    m_patch = mpatches.Patch(color='magenta', label='RF (107,107)')
    # c_patch = mpatches.Patch(color='cyan', label='Level 1 seed 1')
    # red_patch = mpatches.Patch(color='red', label='RF (213,213) seed 2')
    blue_patch = mpatches.Patch(color='blue', label='RF (318,318)')
    green_patch = mpatches.Patch(color='green', label='RF (423,423)')
    # yellow_patch = mpatches.Patch(color='yellow', label='RF (427,427) seed 1')
    # plt.legend(handles=[m_patch,c_patch,red_patch, blue_patch,gree_patch])
    plt.legend(handles=[m_patch, blue_patch, green_patch], prop={"size": 15})
    plt.xlabel("$\lambda$", fontsize=15)
    plt.ylabel("$P(\lambda)$", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    # plt.title("Hessian spectre comparisons")
    plt.xlim(-11, 18)
    plt.savefig("paper_plots/Hessian_spectre_resnet50_trained_cifar10.pdf", bbox_inches="tight")

    ############ VGG
    l_file = open("smoothness/vgg19/l_seed_0_rf_level_1.pkl", "rb")
    w_file = open("smoothness/vgg19/w_seed_0_rf_level_1.pkl", "rb")
    l_s0_level1 = pickle.load(l_file)
    w_s0_level1 = pickle.load(w_file)
    w_s0_level1 = torch.tensor(list([w_s0_level1[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()
    l_file = open("smoothness/vgg19/l_seed_0_rf_level_2.pkl", "rb")
    w_file = open("smoothness/vgg19/w_seed_0_rf_level_2.pkl", "rb")
    l_s0_level2 = pickle.load(l_file)
    w_s0_level2 = pickle.load(w_file)
    w_s0_level2 = torch.tensor(list([w_s0_level2[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()
    l_file = open("smoothness/vgg19/l_seed_0_rf_level_3.pkl", "rb")
    w_file = open("smoothness/vgg19/w_seed_0_rf_level_3.pkl", "rb")
    l_s0_level3 = pickle.load(l_file)
    w_s0_level3 = pickle.load(w_file)
    w_s0_level3 = torch.tensor(list([w_s0_level3[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()

    l_file = open("smoothness/vgg19/l_seed_0_rf_level_4.pkl", "rb")
    w_file = open("smoothness/vgg19/w_seed_0_rf_level_4.pkl", "rb")
    l_s0_level4 = pickle.load(l_file)
    w_s0_level4 = pickle.load(w_file)
    w_s0_level4 = torch.tensor(list([w_s0_level4[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()
    support = torch.linspace(-20, 36, 1000000)
    fig, ax = plt.subplots(figsize=(15, 7))
    # fig, ax = plt.subplots()
    density = torchessian.F(support, l_s0_level1.real, w_s0_level1.real, m)
    ax.plot(support.numpy(), density.numpy(), color='m')
    # density = torchessian.f(support,l_s1_level1.real,w_s1_level1.real,m)
    # ax.plot(support.numpy(), density.numpy(), color='c')
    # density = torchessian.f(support, l_s0_level2.real,w_s0_level2.real, m)
    # ax.plot(support.numpy(), density.numpy(), color='r')
    density = torchessian.F(support, l_s0_level3.real, w_s0_level3.real, m)
    ax.plot(support.numpy(), density.numpy(), color='b')
    density = torchessian.F(support, l_s0_level4.real, w_s0_level4.real, m)
    ax.plot(support.numpy(), density.numpy(), color='g')
    # density = torchessian.f(support,l_s0_levelp.real,w_s0_levelp.real,m)
    # ax.plot(support.numpy(), density.numpy(), color='y')
    ax.set_yscale('log')
    ax.set_yticks([10 ** (i - 7) for i in range(10)])
    ax.set_ylim(10 ** -10, 10 ** 3)
    m_patch = mpatches.Patch(color='magenta', label='RF (181,181)')
    # c_patch = mpatches.patch(color='cyan', label='level 1 seed 1')
    # red_patch = mpatches.patch(color='red', label='rf (359,359) seed 0')
    blue_patch = mpatches.Patch(color='blue', label='RF (537,537)')
    green_patch = mpatches.Patch(color='green', label='RF (715,715)')
    # yellow_patch = mpatches.patch(color='yellow', label='level 4 seed 1')
    # plt.legend(handles=[m_patch,c_patch,red_patch, blue_patch,gree_patch])
    plt.xlim(-20, 36)
    plt.legend(handles=[m_patch, blue_patch, green_patch], prop={"size": 15})
    # plt.title("hessian spectre comparisons")
    plt.xlabel("$\lambda$", fontsize=15)
    plt.ylabel("$P(\lambda)$", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    plt.savefig("paper_plots/Hessian_spectre_vgg_trained_cifar10.pdf", bbox_inches="tight")

    ### Initialisation

    ############ ResNet
    # Level 1
    l_file = open("smoothness/resnet50/cifar10/l_seed_0_rf_level_1_init.pkl", "rb")
    w_file = open("smoothness/resnet50/cifar10/w_seed_0_rf_level_1_init.pkl", "rb")
    l_s0_level1 = pickle.load(l_file)
    w_s0_level1 = pickle.load(w_file)
    w_s0_level1 = torch.tensor(list([w_s0_level1[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()
    #
    # Level 2

    l_file = open("smoothness/resnet50/cifar10/l_seed_0_rf_level_2_init.pkl", "rb")
    w_file = open("smoothness/resnet50/cifar10/w_seed_0_rf_level_2_init.pkl", "rb")
    l_s0_level2 = pickle.load(l_file)
    w_s0_level2 = pickle.load(w_file)
    w_s0_level2 = torch.tensor(list([w_s0_level2[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()

    # Level 3


    l_file = open("smoothness/resnet50/cifar10/l_seed_0_rf_level_3_init.pkl", "rb")
    w_file = open("smoothness/resnet50/cifar10/w_seed_0_rf_level_3_init.pkl", "rb")
    l_s0_level3 = pickle.load(l_file)
    w_s0_level3 = pickle.load(w_file)
    w_s0_level3 = torch.tensor(list([w_s0_level3[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()

    # Level 4

    l_file = open("smoothness/resnet50/cifar10/l_seed_0_rf_level_4_init.pkl", "rb")
    w_file = open("smoothness/resnet50/cifar10/w_seed_0_rf_level_4_init.pkl", "rb")
    l_s0_level4 = pickle.load(l_file)
    w_s0_level4 = pickle.load(w_file)
    w_s0_level4 = torch.tensor(list([w_s0_level4[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()

    # max_values=[l_s0_level1.real.max(),l_s0_level2.real.max(),l_s0_level3.real.max(),l_s0_level4.real.max()]
    # min_values=[l_s0_level1.real.min(),l_s0_level2.real.min(),l_s0_level3.real.min(),l_s0_level4.real.min()]
    # min_values_C=[l_s0_level1.imag.max(),l_s1_level2.imag.max(),l_s1_level3.imag.max(),l_s0_level4.imag.max()]
    # max_values_C=[l_s0_level1.imag.min(),l_s0_level2.imag.min(),l_s0_level3.imag.min(),l_s0_level4.imag.min()]
    rf_level = [1, 2, 3, 4]

    # support = torch.linspace(-4000,4000, 10000000 )
    support = torch.linspace(-900000, 90000, 10000000)
    fig, ax = plt.subplots(figsize=(15, 7))
    # fig, ax = plt.subplots()
    density = torchessian.F(support, l_s0_level1.real, w_s0_level1.real, m)
    ax.plot(support.numpy(), density.numpy(), color='m')
    # density = torchessian.f(support,l_s1_level1.real,w_s1_level1.real,m)
    # ax.plot(support.numpy(), density.numpy(), color='c')
    # density = torchessian.F(support, l_s0_level2.real,w_s0_level2.real, m)
    # ax.plot(support.numpy(), density.numpy(), color='r')
    density = torchessian.F(support, l_s0_level3.real, w_s0_level3.real, m)
    ax.plot(support.numpy(), density.numpy(), color='b')
    density = torchessian.F(support, l_s0_level4.real, w_s0_level4.real, m)
    ax.plot(support.numpy(), density.numpy(), color='g')
    # density = torchessian.f(support,l_s0_levelp.real,w_s0_levelp.real,m)
    # ax.plot(support.numpy(), density.numpy(), color='y')
    ax.set_yscale('log')
    ax.set_yticks([10 ** (i - 7) for i in range(10)])
    ax.set_ylim(10 ** -10, 10 ** 3)
    m_patch = mpatches.Patch(color='magenta', label='RF(107,107)')
    # c_patch = mpatches.Patch(color='cyan', label='level 1 seed 1')
    # red_patch = mpatches.Patch(color='red', label='rf (359,359) seed 0')
    blue_patch = mpatches.Patch(color='blue', label='RF (318,318)')
    green_patch = mpatches.Patch(color='green', label='RF (427,427)')
    # plt.legend(handles=[m_patch,c_patch,red_patch, blue_patch,gree_patch])
    plt.legend(handles=[m_patch, blue_patch, green_patch], prop={"size": 15})
    # plt.title("hessian spectre comparisons at initialisation")
    plt.xlabel("$\lambda$", fontsize=15)
    plt.ylabel("$P(\lambda)$", fontsize=15)
    plt.xlim(-90000, 90000)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    plt.savefig("paper_plots/Hessian_spectre_resnet50_init_cifar10.pdf", bbox_inches="tight")

    ############ VGG
    # Level 1
    l_file = open("smoothness/vgg19/l_seed_0_rf_level_1_init_weights.pkl", "rb")
    w_file = open("smoothness/vgg19/w_seed_0_rf_level_1_init_weights.pkl", "rb")
    l_s0_level1 = pickle.load(l_file)
    w_s0_level1 = pickle.load(w_file)
    w_s0_level1 = torch.tensor(list([w_s0_level1[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()
    #
    # Level 2

    l_file = open("smoothness/vgg19/l_seed_0_rf_level_2_init_weights.pkl", "rb")
    w_file = open("smoothness/vgg19/w_seed_0_rf_level_2_init_weights.pkl", "rb")
    l_s0_level2 = pickle.load(l_file)
    w_s0_level2 = pickle.load(w_file)
    w_s0_level2 = torch.tensor(list([w_s0_level2[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()

    # Level 3


    l_file = open("smoothness/vgg19/l_seed_0_rf_level_3_init_weights.pkl", "rb")
    w_file = open("smoothness/vgg19/w_seed_0_rf_level_3_init_weights.pkl", "rb")
    l_s0_level3 = pickle.load(l_file)
    w_s0_level3 = pickle.load(w_file)
    w_s0_level3 = torch.tensor(list([w_s0_level3[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()

    # Level 4

    l_file = open("smoothness/vgg19/l_seed_0_rf_level_4_init_weights.pkl", "rb")
    w_file = open("smoothness/vgg19/w_seed_0_rf_level_4_init_weights.pkl", "rb")
    l_s0_level4 = pickle.load(l_file)
    w_s0_level4 = pickle.load(w_file)
    w_s0_level4 = torch.tensor(list([w_s0_level4[0, i] ** 2 for i in range(m)]))
    w_file.close()
    l_file.close()

    # max_values=[l_s0_level1.real.max(),l_s0_level2.real.max(),l_s0_level3.real.max(),l_s0_level4.real.max()]
    # min_values=[l_s0_level1.real.min(),l_s0_level2.real.min(),l_s0_level3.real.min(),l_s0_level4.real.min()]
    # min_values_C=[l_s0_level1.imag.max(),l_s1_level2.imag.max(),l_s1_level3.imag.max(),l_s0_level4.imag.max()]
    # max_values_C=[l_s0_level1.imag.min(),l_s0_level2.imag.min(),l_s0_level3.imag.min(),l_s0_level4.imag.min()]
    rf_level = [1, 2, 3, 4]

    support = torch.linspace(-2400, 2400, 100000)
    # support = torch.linspace(min(min_values),max(max_values), 1000000 )
    fig, ax = plt.subplots(figsize=(15, 7))
    # fig, ax = plt.subplots()
    density = torchessian.F(support, l_s0_level1.real, w_s0_level1.real, m)
    ax.plot(support.numpy(), density.numpy(), color='m')
    # density = torchessian.f(support,l_s1_level1.real,w_s1_level1.real,m)
    # ax.plot(support.numpy(), density.numpy(), color='c')
    # density = torchessian.F(support, l_s0_level2.real,w_s0_level2.real, m)
    # ax.plot(support.numpy(), density.numpy(), color='r')
    density = torchessian.F(support, l_s0_level3.real, w_s0_level3.real, m)
    ax.plot(support.numpy(), density.numpy(), color='b')
    density = torchessian.F(support, l_s0_level4.real, w_s0_level4.real, m)
    ax.plot(support.numpy(), density.numpy(), color='g')
    # density = torchessian.f(support,l_s0_levelp.real,w_s0_levelp.real,m)
    # ax.plot(support.numpy(), density.numpy(), color='y')
    ax.set_yscale('log')
    ax.set_yticks([10 ** (i - 7) for i in range(10)])
    ax.set_ylim(10 ** -10, 10 ** 3)
    m_patch = mpatches.Patch(color='magenta', label='RF (181,181)')
    # c_patch = mpatches.Patch(color='cyan', label='level 1 seed 1')
    # red_patch = mpatches.Patch(color='red', label='rf (359,359) seed 0')
    blue_patch = mpatches.Patch(color='blue', label='RF (537,537)')
    green_patch = mpatches.Patch(color='green', label='RF (715,715)')
    # plt.legend(handles=[m_patch,c_patch,red_patch, blue_patch,gree_patch])
    plt.legend(handles=[m_patch, blue_patch, green_patch], prop={"size": 15})
    plt.xlabel("$\lambda$", fontsize=10)
    plt.ylabel("$P(\lambda)$", fontsize=10)
    plt.xlim(-2300, 2300)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    # plt.title("hessian spectre comparisons at initialisation")
    plt.savefig("paper_plots/Hessian_spectre_vgg_init_cifar10.pdf", bbox_inches="tight")
    ############################ Pruning R4esults CIFAR10 ##############################################################
    df = pd.read_csv("cifa10_pruning_results.csv", delimiter=",")

    custom_RS_18_RF = [3, 5, 7, 9, 11, 13, 17, 21, 25, 29, 37, 45, 53, 61, 77, 109, 133]
    pytorch_RS_18_RF = [11, 19, 27, 35, 43, 51, 67, 83, 99, 115, 147, 179, 211, 243, 307, 371, 435]
    resnets_rfs = [108, 110, 213, 318, 423, 1415, 1920, 3100]
    vgg_rfs = [180, 181, 359, 537, 715]

    resnet50_ticks = np.array(range(49))
    resnet18_ticks = np.array(range(17))

    df_level5 = pd.read_csv("RF_resnet50_5_cifar10_0.9_one_shot_summary.csv", delimiter=",")
    df_level6 = pd.read_csv("RF_resnet50_6_cifar10_0.9_one_shot_summary.csv", delimiter=",")
    df_level7 = pd.read_csv("RF_resnet50_7_cifar10_0.9_one_shot_summary.csv", delimiter=",")
    RF = []
    RF.extend([resnets_rfs[5]] * len(df_level5))
    print(len(df_level5))
    RF.extend([resnets_rfs[6]] * len(df_level6))
    print(len(df_level6))
    RF.extend([resnets_rfs[7]] * len(df_level7))
    print(len(df_level7))
    all_df = pd.concat([df_level5, df_level6, df_level7], ignore_index=True)
    all_df["Receptive Field"] = RF
    all_df["Model"] = ["resnet50"] * (len(all_df))
    df["Dense Accuracy"] = df["dense test accuracy"]
    df["Pruned Accuracy"] = df["Pruned test accuracy"]
    df["Receptive Field"] = df["Receptive field"]
    new_all_df = pd.concat([all_df, df], ignore_index=True)
    df = new_all_df
    color1 = "crimson"
    color2 = "cornflowerblue"
    df["Accuracy Reduction (Dense-Pruned)"] = df["Dense Accuracy"] - df["Pruned Accuracy"]
    fig, ax = plt.subplots()
    # sns.lineplot(data=df, x="dense test accuracy", y="Pruned test accuracy", hue="Level",err_style="bars", errorbar=("se", 2))
    sns.lineplot(data=df, x="Receptive Field", y="Dense Accuracy", color=color1, style="Model", err_style="bars",
                 errorbar=("se", 2), legend=False, ax=ax)
    ax2 = plt.twinx()
    # sns.lineplot(data=df.column2, color="b", ax=ax2)
    sns.lineplot(data=df, x="Receptive Field", y="Accuracy Reduction (Dense-Pruned)", color=color2, style="Model",
                 err_style="bars", errorbar=("se", 2), legend=False, ax=ax2)
    # Second legend
    line_solid = mlines.Line2D([], [], color='black', linestyle='-', linewidth=1.5, label=r'ResNet50')
    line_dashed = mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.5, label=r'VGG')
    plt.legend(loc="center right", handles=[line_solid, line_dashed], prop={"size": 12})
    ax.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)
    ax.set_ylabel("Dense Accuracy", color=color1, size=12)
    ax2.set_ylabel("$\Delta$", color=color2, size=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel("Receptive Field", size=12)
    plt.savefig("paper_plots/Pruned_results_CIFAR10.pdf", bbox_inches="tight")
    ############################ Pruning Results Tiny Imagenet ##############################################################


    df_vgg = pd.read_csv("vgg_19_tiny_imagenet.csv", delimiter=",")
    df_vgg["Receptive Field"] = df_vgg["Receptive field"]
    df_resnet50 = pd.read_csv("resnet50_tiny_imagenet_pruning_summary.csv", delimiter=",")
    df_resnet50["Receptive Field"] = df_resnet50["Receptive field"]
    df_level5 = pd.read_csv("RF_resnet50_5_tiny_imagenet_0.9_one_shot_summary.csv", delimiter=",")
    df_level6 = pd.read_csv("RF_resnet50_6_tiny_imagenet_0.9_one_shot_summary.csv", delimiter=",")
    df_level7 = pd.read_csv("RF_resnet50_7_tiny_imagenet_0.9_one_shot_summary.csv", delimiter=",")
    RF = []
    RF.extend([resnets_rfs[5]] * len(df_level5))
    print(len(df_level5))
    RF.extend([resnets_rfs[6]] * len(df_level6))
    print(len(df_level6))
    RF.extend([resnets_rfs[7]] * len(df_level7))
    print(len(df_level7))
    all_df = pd.concat([df_level5, df_level6, df_level7], ignore_index=True)
    all_df["Receptive Field"] = RF
    all_resnet = pd.concat([df_resnet50, all_df], ignore_index=True)
    all_resnet["Model"] = ["resnet50"] * len(all_resnet)
    df_vgg["Model"] = ["vgg"] * len(df_vgg)
    df = pd.concat([all_resnet, df_vgg], ignore_index=True)

    color1 = "crimson"
    color2 = "cornflowerblue"
    df["Accuracy Reduction (Dense-Pruned)"] = df["Dense Accuracy"] - df["Pruned Accuracy"]
    fig, ax = plt.subplots()
    # sns.lineplot(data=df, x="dense test accuracy", y="Pruned test accuracy", hue="Level",err_style="bars", errorbar=("se", 2))
    sns.lineplot(data=df, x="Receptive Field", y="Dense Accuracy", color=color1, style="Model", err_style="bars",
                 errorbar=("se", 2), legend=False, ax=ax)
    ax2 = plt.twinx()
    # sns.lineplot(data=df.column2, color="b", ax=ax2)
    sns.lineplot(data=df, x="Receptive Field", y="Accuracy Reduction (Dense-Pruned)", color=color2, style="Model",
                 err_style="bars", errorbar=("se", 2), legend=False, ax=ax2)
    # Second legend
    line_solid = mlines.Line2D([], [], color='black', linestyle='-', linewidth=1.5, label=r'ResNet50')
    line_dashed = mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.5, label=r'VGG')
    plt.legend(loc="center right", handles=[line_solid, line_dashed], prop={"size": 12})
    ax.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)
    ax.set_ylabel("Dense Accuracy", color=color1, size=12)
    ax2.set_ylabel("$\Delta$", color=color2, size=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel("Receptive Field", size=12)
    plt.savefig("paper_plots/Pruned_results_TinyImageNet.pdf", bbox_inches="tight")

if __name__ == '__main__':
    main()