#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Dimitrios Stoidis, dimitrios.stoidis@qmul.ac.uk
# - Alessio Xompero, a.xompero@qmul.ac.uk
# - Myriam Bontonou, myriam.bontonou@ens-lyon.fr
#
#
#  Created Date: 2023/01/17
# Modified Date: 2024/02/19
#
# MIT License

# Copyright (c) 2024 GraphNEx

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import os
import sys
import argparse

import torch
import random
from sklearn.metrics import confusion_matrix
import csv

import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pdb import set_trace as bp


# ----------------------------------------------------------------------------
# CONSTANTS
#
device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
def set_seed(seed_val) -> None:
    """ """
    # seed_everything(seed_val)
    #
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    np.random.seed(seed_val)
    random.seed(seed_val)

    torch.manual_seed(seed_val)

    # if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch.use_deterministic_algorithms(True)

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed_val)
    print(f"Random seed set as {seed_val}")


def save_model(net, measure, outdir, filename, mode="last", epoch=-1):
    """

    Arguments:
        - measure
        - epoch
        - outdir
        - filename
        - mode
    """
    assert mode in ["best", "last"]

    # Prepare state to save the model via PyTorch
    state = {
        "net": net.state_dict(),
        "measure": measure,
        "epoch": epoch,
        "type": mode,
    }

    pathfilename = os.path.join(outdir, mode + "_" + filename)

    # Save the mode at the given path with given filename
    torch.save(state, pathfilename)


def print_model_parameters(net):
    """Print to screen the total number of parameters of a model.

    The functions displays both the optimised parameters and the trainable
    parameters (including those that have been fixed) of a model provided as
    input.

    Arguments:
        - net: the input model.
    """
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total optimised parameters: {}".format(param_num))

    param_num = sum(p.numel() for p in net.parameters())
    print("Total trainable parameters: {}".format(param_num))


def check_if_square(A):
    m, n = A.shape
    return m == n


def check_if_symmetric(A, therr=1e-8):
    return np.all(np.abs(A - A.T) < therr)


def crossval_stats_summary(in_filename, out_filename, measure):
    """Output the statistics of a model trained with K-Fold Cross-Validation.

    The function reads the performance measures of a model trained on a given
    dataset using the stratified K-Fold Cross-Validation strategy and computes
    min, max, average and standard deviation for each performance measure.

    The statistics are outputted into a .csv file specific for the model and
    in the directory where the trained model is saved, e.g.,
    /trained_models/biology/brca/mlp_crossval_stats.csv.

    The performance measures are:
        * UBA-T: (unbalanced) accuracy in the training split.
        * UBA-V: (unbalanced) accuracy in the validation split.
        * BA-T: balanced accuracy in the training split.
        * BA-V: balanced accuracy in the validation split.
        * wF1-T: weighted F1-score in the training split.
        * wF1-V: weighted F1-score in the validation split.
        * mF1-T: macro F1-score in the training split.
        * mF1-V: macro F1-score in the validation split.

    The input is a .txt file with 10 columns separated by a tab. The first two
    columns are fold id and epoch (best epoch where the model was saved using
    early stopping). The last 8 columns are the performance measures defined
    above.
    """
    assert measure in [
        "precision",
        # "recall",
        "accuracy",
        "balanced_accuracy",
        "weighted_f1_score",
        "macro_f1_score",
    ]

    if measure == "precision":
        col_str = "P-V"
    # elif measure == "recall":
    # col_str = "R-V"
    elif measure == "accuracy":
        col_str = "UBA-V"
    elif measure == "balanced_accuracy":
        col_str = "BA-V"
    elif measure == "weighted_f1_score":
        col_str = "wF1-V"
    elif measure == "macro_f1_score":
        col_str = "MF1-V"

    df = pd.read_csv(in_filename, sep="\t", index_col=False)

    idxmin = np.where(df[col_str] == df[col_str].min())[0]
    idxmax = np.where(df[col_str] == df[col_str].max())[0]

    headers = [
        "measure",
        "P-T",
        "P-V",
        # "R-T",
        # "R-V",
        "UBA-T",
        "UBA-V",
        "BA-T",
        "BA-V",
        "wF1-T",
        "wF1-V",
        "MF1-T",
        "MF1-V",
    ]

    df_out = pd.DataFrame(columns=headers)
    df_out.loc[0] = ["min"] + df.iloc[idxmin[0], 2:].values.tolist()
    df_out.loc[1] = ["max"] + df.iloc[idxmax[0], 2:].values.tolist()
    df_out.loc[2] = ["avg"] + df.iloc[:, 2:].mean().values.tolist()
    df_out.loc[3] = ["std"] + df.iloc[:, 2:].std().values.tolist()

    df_out.to_csv(out_filename, header=headers, index=False)


def plot_learning_curves(outdir, infilename, outfilename, measure, mode="final"):
    """

    Arguments:
        - outdir: directory where the txt file with the learning curves was saved.
        - infilename: name of the txt file with the learning curves
        - measure: performance measure or to plot
        - mode: mode in which the model was training
    """
    assert mode in ["crossval", "final", "original"]

    assert measure in [
        "loss",
        "precision",
        # "recall",
        "accuracy",
        "balanced_accuracy",
        "weighted_f1_score",
        "macro_f1_score",
    ]

    # Read file
    df = pd.read_csv("{}/{}".format(outdir, infilename), sep="\t", index_col=False)

    # Select the epochs to plot as x-axis
    x = df["Epoch"].values

    # Select measure
    if measure == "loss":
        col_str = "loss"
    elif measure == "precision":
        col_str = "P"
    # elif measure == "recall":
    # col_str = "R"
    elif measure == "accuracy":
        col_str = "UBA"
    elif measure == "balanced_accuracy":
        col_str = "BA"
    elif measure == "weighted_f1_score":
        col_str = "wF1"
    elif measure == "macro_f1_score":
        col_str = "MF1"

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if mode == "final":
        if measure == "loss":
            y = df[col_str].values
        else:
            y = df[col_str + "-T"].values

        # plot the data
        ax.plot(x, y, color="tab:blue")

        # set the limits
        ax.set_xlim([0, np.max(x)])

        if measure == "loss":
            ax.set_ylim([0, np.max(y) * 1.2])
        else:
            ax.set_ylim([np.max((0, np.min(y) * 0.8)), np.max(y) * 1.2])

    else:
        # if measure == "loss":
        #     y1 = df[col_str].values
        #     y2 = df[col_str].values
        # else:
        #     y1 = df[col_str + "-T"].values
        #     y2 = df[col_str + "-V"].values

        y1 = df[col_str + "-T"].values
        y2 = df[col_str + "-V"].values

        # plot the data
        ax.plot(x, y1, color="tab:blue")
        ax.plot(x, y2, color="tab:orange")

        # set the limits
        max_y = np.max((np.max(y1), np.max(y2)))
        ax.set_xlim([0, np.max(x)])

        if measure == "loss":
            ax.set_ylim([0, max_y * 1.2])
        else:
            ax.set_ylim([np.max((0, max_y * 0.8)), max_y * 1.2])

    ax.set_title("Learning curves")

    # plt.savefig(fig, format="png")
    plt.savefig(os.path.join(outdir, outfilename), bbox_inches="tight")


def convert_adj_to_edge_index(adjacency_matrix):
    """
    Taken from https://github.com/gordicaleksa/pytorch-GAT/blob/main/utils/utils.py

    Handles both adjacency matrices as well as connectivity masks used in softmax (check out Imp2 of the GAT model)
    Connectivity masks are equivalent to adjacency matrices they just have -inf instead of 0 and 0 instead of 1.
    I'm assuming non-weighted (binary) adjacency matrices here obviously and this code isn't meant to be as generic
    as possible but a learning resource.
    """
    assert isinstance(
        adjacency_matrix, np.ndarray
    ), f"Expected NumPy array got {type(adjacency_matrix)}."
    height, width = adjacency_matrix.shape
    assert height == width, f"Expected square shape got = {adjacency_matrix.shape}."

    # If there are infs that means we have a connectivity mask and 0s are where the edges in connectivity mask are,
    # otherwise we have an adjacency matrix and 1s symbolize the presence of edges.
    active_value = 0 if np.isinf(adjacency_matrix).any() else 1

    edge_index = []
    for src_node_id in range(height):
        for trg_nod_id in range(width):
            if adjacency_matrix[src_node_id, trg_nod_id] == active_value:
                edge_index.append([src_node_id, trg_nod_id])

    return np.asarray(edge_index).transpose()  # change shape from (N,2) -> (2,N)


def convert_adj_to_edge_index_weight(adjacency_matrix, edge_th=0):
    """
    Taken from https://github.com/gordicaleksa/pytorch-GAT/blob/main/utils/utils.py

    Handles both adjacency matrices as well as connectivity masks used in softmax (check out Imp2 of the GAT model)
    Connectivity masks are equivalent to adjacency matrices they just have -inf instead of 0 and 0 instead of 1.
    I'm assuming non-weighted (binary) adjacency matrices here obviously and this code isn't meant to be as generic
    as possible but a learning resource.
    """
    assert isinstance(
        adjacency_matrix, np.ndarray
    ), f"Expected NumPy array got {type(adjacency_matrix)}."
    height, width = adjacency_matrix.shape
    assert height == width, f"Expected square shape got = {adjacency_matrix.shape}."

    # If there are infs that means we have a connectivity mask and 0s are where the edges in connectivity mask are,
    # otherwise we have an adjacency matrix and 1s symbolize the presence of edges.
    active_value = 0 if np.isinf(adjacency_matrix).any() else 1

    edge_index = []
    weight_list = []
    for src_node_id in range(height):
        for trg_nod_id in range(width):
            if adjacency_matrix[src_node_id, trg_nod_id] > edge_th:
                edge_index.append([src_node_id, trg_nod_id])
                weight_list.append([adjacency_matrix[src_node_id, trg_nod_id]])

    # change shape from (N,2) -> (2,N)
    return np.asarray(edge_index).transpose(), np.asarray(
        weight_list
    ).transpose().astype(np.float32)


def convert_adj_list_to_edge_index(adj_list, b_undirected=False):
    """ """
    cnt = 0

    edge_index = []
    for key, value in adj_list.items():
        if len(value) == 0:
            continue
        else:
            for v in value:
                if type(v) == list:
                    edge_index.append([int(key), v[0]])

                    if b_undirected:
                        edge_index.append([v[0], int(key)])

                else:
                    edge_index.append([int(key), v])

                    if b_undirected:
                        edge_index.append([v, int(key)])

    return np.array(edge_index).squeeze().transpose()


def convert_adj_list_to_edge_index_and_edge_weight(adj_list, b_undirected=False):
    cnt = 0

    edge_index = []
    edge_weight = []
    for key, value in adj_list.items():
        if len(value) == 0:
            continue
        else:
            for v in value:
                if type(v) == list:
                    edge_index.append([int(key), v[0]])
                    edge_weight.append(v[1])
                    if b_undirected:
                        edge_index.append([v[0], int(key)])
                        edge_weight.append(v[1])
                else:
                    edge_index.append([int(key), v])
                    edge_weight.append(1)
                    if b_undirected:
                        edge_index.append([v, int(key)])
                        edge_weight.append(1)

    return np.array(edge_index).squeeze().transpose(), np.array(edge_weight)


# ----------------------------------------------------------------------------
def load_coco_classes(root_dir):
    """
    Loads class labels at 'path'
    Code originally from https://pjreddie.com/darknet/yolo/.
    """
    fp = open(os.path.join(root_dir, "resources", "coco.names"), "r")
    names = fp.read().split("\n")[:-1]
    return names


##############################################################################
def update_config_file(config, args):
    """ """
    if args.seed is not None:
        config["paths"]["seed"] = args.seed

    config["dataset"] = args.dataset

    # set_seed(config["params"]["seed"])

    # Add dataset configurations
    with open(os.path.join("configs", "datasets.json")) as f:
        data_config = json.load(f)

    config["paths"] = data_config["paths"]
    config["datasets"] = data_config["datasets"]

    # if args.root_dir is not None:
    #     config["paths"]["root_dir"] = args.root_dir

    # if args.data_dir is not None:
    #     config["paths"]["data_dir"] = args.data_dir

    if args.training_mode is not None:
        config["params"]["training_mode"] = args.training_mode

    if args.split_mode is not None:
        config["params"]["split_mode"] = args.split_mode

    if args.fold_id is not None:
        config["params"]["fold_id"] = args.fold_id

    if args.graph_mode is not None:
        config["net_params"]["graph_mode"] = args.graph_mode

    config["category_mode"] = -1  # default

    return config


def extend_parser_training(parser):
    """Extend with TRAINING parameters"""
    #
    parser.add_argument("--batch_sz_train", type=int)
    parser.add_argument("--batch_sz_val", type=int)

    parser.add_argument("--num_epochs", type=int)

    parser.add_argument(
        "--optim",
        type=str,
        help="optimizer to use (default: Adam)",
    )
    parser.add_argument("--learning_rate", type=float)

    # Resuming training parameters
    parser.add_argument(
        "--measure",
        type=str,
        choices=[
            "balanced_accuracy",
            "accuracy",
            "macro_f1_score",
            "weighted_f1_score",
        ],
        help="measure to use for the resume from checkpoint",
    )

    if sys.version_info[0] >= 3:
        if sys.version_info[1] < 9:
            parser.add_argument(
                "--weight_loss",
                action="store_true",
                help="Force to use a weighted loss to handle imbalance dataset.",
            )
            parser.add_argument(
                "--resume",
                "-r",
                action="store_true",
                help="resume from checkpoint",
            )
            parser.set_defaults(feature=False)
        else:
            parser.add_argument("--weight_loss", action=argparse.BooleanOptionalAction)
            parser.add_argument("--resume", action=argparse.BooleanOptionalAction)
    else:
        parser.add_argument(
            "--weight_loss",
            action="store_true",
            help="Force to use a weighted loss to handle imbalance dataset.",
        )
        parser.add_argument(
            "--no-weight_loss", dest="weight_loss", action="store_false"
        )
        parser.add_argument(
            "--resume",
            "-r",
            action="store_true",
            help="resume from checkpoint",
        )
        parser.add_argument("--no-resume", dest="resume", action="store_false")
        parser.set_defaults(feature=False)

    return parser


def extend_parser_testing(parser):
    """ """
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--res_dir", type=str)
    parser.add_argument("--batch_size", type=int)

    return parser


def extend_parser_explaining(parser):
    # Paths
    parser.add_argument("--model_dir", type=str, default=".")
    parser.add_argument("--res_dir", type=str, default=".")

    # Name of the saved model
    parser.add_argument("--model_filename", type=str, default=".")

    # Dataset
    parser.add_argument(
        "--partition",
        type=str,
        choices=["train", "test"],
        help="Set on which the scores are computed.",
    )
    parser.add_argument("--batch_size", type=int, default=1)

    # XAI method
    parser.add_argument(
        "--method", type=str, choices=["Integrated_Gradients", "Kernel_SHAP"]
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=100,
        help="Used if method == Integrated_Gradients.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Used if method == Kernel_Shap.",
    )
    parser.add_argument(
        "--studied_class",
        type=int,
        nargs="+",
        help="Compute the scores for all examples belonging to a class in `studied_class`.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["null", "avg"],
        help="Example used as a reference to compute the explainability scores. `null` is a zero tensor. `avg` is the average of the training examples not in `studied_class`.",
    )

    return parser


def GetParser(desc="", mode="train"):
    """ """
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Seed
    parser.add_argument("--seed", type=int)

    # Paths
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--data_dir", type=str)

    parser.add_argument("--n_workers", type=int)

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["PrivacyAlert"],
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        choices=["final", "crossval", "original"],
        required=False,
        help="Choose to run K-fold cross-validation or train the final model (full training set without validation split)",
    )

    parser.add_argument(
        "--config", required=True, help="Please provide a config.json file"
    )

    parser.add_argument(
        "--mode",
        required=True,
        help="Please provide either training, testing or explaining mode.",
        choices=["training", "testing", "explaining"],
    )

    # Model
    parser.add_argument("--n_out_classes", type=int)

    parser.add_argument("--fold_id", type=int)

    if sys.version_info[0] >= 3:
        if sys.version_info[1] < 9:
            parser.add_argument(
                "--use_bce",
                action="store_true",
                help="Force to use binary cross-entropy.",
            )
            parser.set_defaults(feature=False)
        else:
            parser.add_argument("--use_bce", action=argparse.BooleanOptionalAction)
    else:
        parser.add_argument(
            "--use_bce",
            action="store_true",
            help="Force to use binary cross-entropy.",
        )
        parser.add_argument("--no-use_bce", dest="use_bce", action="store_false")
        parser.set_defaults(feature=False)

    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--graph_mode",
        type=str,
        choices=["bipartite", "co_occ_and_bip", "co_occ_then_bip"],
    )

    # Only for testing
    parser.add_argument("--model_mode", type=str, choices=["best", "last"])

    return parser
