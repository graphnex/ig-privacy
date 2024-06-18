#!/usr/bin/env python
#
# <Brief description here>
#
################################################################################
# Authors:
# - Alessio Xompero
# - Myriam Bontonou
#
# Email: a.xompero@qmul.ac.uk
#
#  Created Date: 2023/02/09
# Modified Date: 2024/02/13
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
import inspect
import argparse

from datetime import datetime

# setting path
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.insert(0, pp_dir)

import numpy as np
import pandas as pd

import json
import csv

from srcs.perfmeas_tracker import PerformanceMeasuresTracker
from srcs.datasets.wrapper import WrapperDatasets

from pdb import set_trace as bp

# ----------------------------------------------------------------------------


class EvaluationToolkit:
    """ """

    def __init__(self, args):
        self.dataset = args.dataset

        self.model_results_csv = args.model_results_csv
        
        self.model_name = args.model_name
        self.model_mode = args.model_mode

        self.out_file = args.out_file

        self.repo_dir = args.root_dir
        self.data_dir = args.data_dir

        self.n_cls = args.n_out_classes

        self.partition = args.training_mode
        self.split_mode = args.split_mode
        self.fold_id = args.fold_id

        self.perf_measures = PerformanceMeasuresTracker(
            dataset=self.dataset, n_cls=self.n_cls
        )
        self.perf_measures.set_beta(args.beta)

        self.load_annotations(args.b_filter_imgs)

    def load_annotations(self, _b_filter_imgs=False):
        """Load into memory the labels of the training set."""

        data_wrapper = WrapperDatasets(
            root_dir=self.repo_dir,
            data_dir=self.data_dir,
            num_classes=self.n_cls,
            # n_graph_nodes=self.n_graph_nodes,
            # node_feat_size=self.node_feature_size,
            fold_id=self.fold_id,
            # adj_mat_fn=None,
            # graph_mode=net_params["graph_type"],
        )

        data_wrapper.load_split_set(
            self.dataset,
            partition=self.partition,
            mode="train" if self.split_mode == "val" else self.split_mode,
            b_filter_imgs=_b_filter_imgs,
        )

        data_split = data_wrapper.get_data_split(self.split_mode)
        self.annotations = np.array(data_split.get_labels())

    def get_headers_binary(self):
        headers = [
            "date",
            "model name",
            "model_mode",
            "dataset",
            "partition",
            "fold",
            "split",
            "P_0",
            "R_0",
            "F1_0",
            "P_1",
            "R_1",
            "F1_1",
            "P",
            "R (BA)",
            "ACC",
            "MF1",
            "wF1",
            # "BA",
            # "Beta_F1",
        ]

        return headers

    def get_model_res_binary(self):
        model_res = [
            datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            self.model_name,
            self.model_mode,
            self.dataset,
            self.partition,
            self.fold_id,
            self.split_mode,
            self.perf_measures.get_measure("precision_0"),
            self.perf_measures.get_measure("recall_0"),
            self.perf_measures.get_measure("f1_score_0"),
            self.perf_measures.get_measure("precision_1"),
            self.perf_measures.get_measure("recall_1"),
            self.perf_measures.get_measure("f1_score_1"),
            self.perf_measures.get_measure("precision"),
            self.perf_measures.get_measure("recall"),
            self.perf_measures.get_measure("accuracy"),
            self.perf_measures.get_measure("macro_f1_score"),
            self.perf_measures.get_measure("weighted_f1_score"),
            # self.perf_measures.get_measure("balanced_accuracy"),
            # self.perf_measures.get_measure("beta_f1_score"),
        ]

        return model_res

    def get_headers(self):
        if self.n_cls == 2:
            headers = self.get_headers_binary()

        return headers

    def get_model_res(self):
        if self.n_cls == 2:
            model_res = self.get_model_res_binary()

        return model_res

    def save_model_res_to_csv(self, m_res):
        """ """
        model_res = []
        for x in m_res[:7]:
            model_res.append(x)

        for x in m_res[7:]:
            model_res.append("{:.2f}".format(x * 100))  # make performance in percentages

        if os.path.exists(self.out_file):
            fh = open(self.out_file, "a")

            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = csv.writer(fh)

            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(model_res)

            # Close the file object
            fh.close()

        else:
            headers = self.get_headers()

            fh = open(self.out_file, "w")

            writer_object = csv.writer(fh)

            writer_object.writerow(headers)
            writer_object.writerow(model_res)

            # Close the file object
            fh.close()

    def run(self):
        """ """
        # Read comma-based CSV file where the model predictions are saved
        es = pd.read_csv(self.model_results_csv, sep=",", index_col=False)

        preds = es["pred_class"].values

        self.perf_measures.compute_all_metrics(self.annotations, preds)

        model_res = self.get_model_res()

        self.save_model_res_to_csv(model_res)

        print("Performance metrics saved in " + self.out_file)


def GetParser():
    parser = argparse.ArgumentParser(
        description="GraphNEx Evaluation Toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model_results_csv", default="random.csv", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["PrivacyAlert"],
        required=True,
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_mode", type=str, required=True, choices=["best","last"])
    parser.add_argument("--beta", default=2.0, type=float)

    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--root_dir", default=".", type=str)
    parser.add_argument("--data_dir", default=".", type=str)

    parser.add_argument("--n_out_classes", default=2, type=int, choices=[2])

    parser.add_argument(
        "--training_mode",
        type=str,
        choices=["final", "crossval", "original"],
        required=True,
        help="Choose to run K-fold cross-validation or train the final model (full training set without validation split)",
    )
    parser.add_argument("--fold_id", type=int)

    parser.add_argument(
        "--split_mode",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Choose the data split for the evaluation (training, validation, testing)!",
    )

    if sys.version_info[0] >= 3:
        if sys.version_info[1] < 9:
            parser.add_argument(
                "--b_filter_imgs",
                action="store_true",
                help="Force to use binary cross-entropy.",
            )
            parser.set_defaults(feature=False)
        else:
            parser.add_argument(
                "--b_filter_imgs", action=argparse.BooleanOptionalAction
            )
    else:
        parser.add_argument(
            "--b_filter_imgs",
            action="store_true",
            help="Force to use binary cross-entropy.",
        )
        parser.add_argument(
            "--no-b_filter_imgs", dest="b_filter_imgs", action="store_false"
        )
        parser.set_defaults(feature=False)

    return parser


if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))

    # Arguments
    parser = GetParser()
    args = parser.parse_args()

    eval_toolkit = EvaluationToolkit(args)
    eval_toolkit.run()
