#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/03/09
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
# FROM, # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import inspect
import os
import sys
import argparse  # Parser for command-line options, arguments and sub-commands
import json  # Open standard file format to store data

# setting path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.insert(0, pp_dir)

from srcs.prior_graph.prior_graph_gip import PriorKnowlegeGraphGIP
from srcs.prior_graph.prior_graph_gpa import PriorKnowlegeGraphGPA
from srcs.prior_graph.objects_only import PriorGraphObjectsOnly


#############################################################################


def GetParser():
    parser = argparse.ArgumentParser(
        description="Prior Knowledge Graph Builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--root_dir", type=str)

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["PrivacyAlert"],
        # default="IPD",
    )

    parser.add_argument("--scene_probs_fn", type=str, default="scene_probs.csv")

    parser.add_argument(
        "--model_name",
        type=str,
        default="obj_scene",
        choices=["gip", "gpa", "obj_scene", "obj-only"],
    )

    parser.add_argument("--n_obj_cats", type=int)
    parser.add_argument("--n_scene_cats", type=int)
    parser.add_argument("--n_privacy_cls", type=int, default=2)

    parser.add_argument(
        "--category_mode",
        type=int,
        default=-1,
        choices=[-1, 0, 1, 2],
        help="Choose the prior graph to build: using all privacy classes (-1), using only the private class (0), or using only the public class (1).",
    )

    parser.add_argument("-k", type=int, default=5)

    parser.add_argument(
        "--fold_id",
        type=int,
        # default=0,
        # choices=[0, 1, 2, 99]
    )

    parser.add_argument("--partitions_fn", type=str, default="labels_splits.csv")

    parser.add_argument(
        "--self_edges",
        default=True,
        type=bool,
        help="include self-edges for object categories in the graph",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="compute",
        choices=["compute", "stats", "compute_stats"],
    )

    parser.add_argument(
        "--training_mode",
        type=str,
        default="final",
        choices=["final", "crossval", "original"],
        help="Choose to run K-fold cross-validation or train the final model (full training set without validation split)",
    )

    parser.add_argument(
        "--config", required=True, help="Please provide a config.json file"
    )

    return parser


if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))

    # Arguments
    parser = GetParser()
    args = parser.parse_args()

    with open(os.path.join("configs", args.config)) as f:
        config = json.load(f)

    config["category_mode"] = args.category_mode

    if args.n_scene_cats is not None:
        config["net_params"]["num_scene_cat"] = args.n_scene_cats

    if args.dataset is not None:
        config["dataset"] = args.dataset

    if args.model_name == "gip":
        g_builder = PriorKnowlegeGraphGIP(args)

        if args.mode == "compute":
            g_builder.run_compute_graph()
        elif args.mode == "stats":
            g_builder.run_graph_analysis()
        elif args.mode == "compute_stats":
            g_builder.run_compute_graph()
            g_builder.run_graph_analysis()

    elif args.model_name == "gpa":
        g_builder = PriorKnowlegeGraphGPA(config)

        if args.mode == "compute":
            g_builder.run_compute_graph()
        elif args.mode == "stats":
            g_builder.run_graph_analysis()
        elif args.mode == "compute_stats":
            g_builder.run_compute_graph()
            g_builder.run_graph_analysis()

    elif args.model_name == "obj-only":
        g_builder = PriorGraphObjectsOnly(config)

        if args.mode == "compute":
            g_builder.run_compute_graph()
