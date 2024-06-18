#!/usr/bin/env python
#
# Python class for training Graph Privacy Advisor.
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
# - Myriam Bontonou, myriam.bontonou@ens-lyon.fr
#
#  Created Date: 2023/09/21
# Modified Date: 2024/02/27
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
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------------
# Libraries
import os
import sys
import inspect
import argparse

import json

import torch

# Path of G-architecture
current_path = os.path.abspath(inspect.getfile(inspect.currentframe()))

dirlevel1 = os.path.dirname(current_path)
dirlevel0 = os.path.dirname(dirlevel1)

print(dirlevel0)

sys.path.insert(0, dirlevel0)

# Package modules
from srcs.prior_graph.objects_only import PriorGraphObjectsOnly

from srcs.load_net import gnn_model

from srcs.trainer_base import TrainerBaseClass
from srcs.tester_base import TesterBaseClass
from srcs.explainer_base import ExplainerBaseClass

from srcs.utils import (
    device,
    GetParser,
    print_model_parameters,
    update_config_file,
    set_seed
)

from pdb import set_trace as bp  # This is only for debugging


#############################################################################
def get_prior_graph(config, data_wrapper):
    """ """
    try:
        if config["net_params"]["graph_type"] == "obj-only":
            prior_graph = PriorGraphObjectsOnly(config)
        else:
            raise ValueError('Prior graph type not valid!')
    except  (ValueError, IndexError):
        exit('Could not complete request.')

    prior_graph.load_adjacency_matrix(
        data_wrapper.get_data_dir(),
        "gpa",
        config["params"]["training_mode"],
        config["params"]["fold_id"],
        "ac",
        self_edges=False,
        a_th=config["net_params"]["prior_graph_thresh"]
    )

    return prior_graph

#############################################################################

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
            parser.add_argument(
                "--weight_loss", action=argparse.BooleanOptionalAction
            )
            parser.add_argument(
                "--resume", action=argparse.BooleanOptionalAction
            )
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

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_filename", type=str)

    parser.add_argument("--batch_size", type=int)

    return parser

##############################################################################

#
class Trainer(TrainerBaseClass):
    def __init__(self, config, args):
        super().__init__(config, args)

        # if config["net_name"] in ["GCN", "GAT"]:
        #     self.adjacency_filename = get_prior_graph_filename(config)
        # else:
        #     self.adjacency_filename = None

        self.load_training_graph_data(
            config["dataset"], config["net_params"], config["b_filter_imgs"]
        )
        self.initialise_performance_trackers(
            config["dataset"], self.n_out_classes
        )

        self.net = gnn_model(config["net_name"], config)

        if config["net_name"] in ["GPA", "GIP"]:
            prior_graph = get_prior_graph(config, self.data_wrapper)
            self.net.initialise_prior_graph(prior_graph)
        else:
            self.adjacency_filename = None

        print("\n{:s} parameters: ".format(config["net_name"]))
        print_model_parameters(self.net)

        self.configure_optimizer(self.net, config)

        self.initialise_checkpoint_dir(
            self.net.get_model_name(), self.n_out_classes
        )
        self.initialise_log(self.net.get_model_name(), self.n_out_classes)


class Tester(TesterBaseClass):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.load_testing_graph_data(
            config["params"]["training_mode"],
            config["dataset"],
            config["net_params"],
            b_filter_imgs=config["b_filter_imgs"],
            split_mode=args.split_mode,
        )

        # Model network
        self.net = gnn_model(config["net_name"], config)

        if config["net_name"] in ["GPA"]:
            prior_graph = get_prior_graph(config, self.data_wrapper)
            self.net.initialise_prior_graph(prior_graph)

            if config["net_name"] == "GPA":
                self.net.set_batch_size(config["params"]["batch_size"])
        else:
            self.adjacency_filename = None

        print("\n{:s} parameters: ".format(config["net_name"]))
        print_model_parameters(self.net)

        if config["params"]["training_mode"] == "final":
            prefix_net = "last_acc_"
        else:
            prefix_net = args.model_mode + "_acc_"

        checkpoint = self.get_model(
            self.get_filename(
                self.net.get_model_name(),
                ".pth",  # extension of the models
                prefix=prefix_net,
            ),
            self.net.get_model_name(),
        )

        self.net.load_state_dict(checkpoint["net"])



class Explainer(ExplainerBaseClass):
    def __init__(self, config, args):
        super().__init__(config, args)
        # Graph
        if config["net_name"] in ["GCN", "GAT"]:
            self.adjacency_filename = get_prior_graph_filename(config)
        else:
            self.adjacency_filename = None

        # Model network
        config["net_params"]["output_proba"] = True
        self.net = gnn_model(config["net_name"], config)
        print("\n{:s} parameters: ".format(config["net_name"]))
        print_model_parameters(self.net)
        if config["params"]["training_mode"] == "final":
            prefix_net = "last_acc_"
        else:
            prefix_net = args.model_mode + "_acc_"
        checkpoint = self.get_model(
            self.get_filename(
                self.net.get_model_name(),
                ".pth",  # extension of the models
                prefix=prefix_net,
            ),
            self.net.get_model_name(),
        )
        self.net.load_state_dict(checkpoint["net"])
        self.net = self.net.to(device)
        self.net.eval()

##############################################################################
class TesterRule(TesterBaseClass):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.adjacency_filename = None

        self.load_testing_graph_data(
            config["params"]["training_mode"],
            config["dataset"],
            config["net_params"],
            b_filter_imgs=config["b_filter_imgs"],
            split_mode=args.split_mode,
        )

        # Model network
        self.net = gnn_model(config["net_name"], config)

        self.checkpoint_dir = os.path.join(
            self.model_dir,
            # "{:d}-class".format(self.n_out_classes),
            config["net_name"],
        )

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)



#############################################################################
#
if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))
    # print("PyTorch {}".format(torch.__version__))
    print("Using {}".format(device))

    # Arguments
    parser = GetParser()

    if parser.parse_args().mode == "training":
        parser = extend_parser_training(parser)

    elif parser.parse_args().mode == "testing":
        parser = extend_parser_testing(parser)

    elif parser.parse_args().mode == "explain":
        parser = extend_parser_explaining(parser)

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    config = update_config_file(config, args)

    set_seed(config["params"]["seed"])

    assert args.mode in ["training", "testing", "explaining"]
    if args.mode == "training":
        processor = Trainer(config, args)
    elif args.mode == "testing":
        if config["net_name"] in ["PersonRule"]:
            processor = TesterRule(config, args)
        else:
            processor = Tester(config, args)
    else:
        processor = Explainer(config, args)

    processor.run()
