#!/usr/bin/env python
#
# Parent/base class for testing different machine learning models.
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2023/05/09
# Modified Date: 2024/02/12
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

import os
import sys
import inspect
import argparse

# setting path
# current_dir = os.path.dirname(
#     os.path.abspath(inspect.getfile(inspect.currentframe()))
# )
# parent_dir = os.path.dirname(current_dir)
# pp_dir = os.path.dirname(parent_dir)
# sys.path.insert(0, pp_dir)

import time
from datetime import datetime

import numpy as np
np.set_printoptions(threshold=sys.maxsize, precision=2)

import pandas as pd

from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Package modules
from srcs.datasets.wrapper import WrapperDatasets
from srcs.utils import device
from srcs.logging_gnex import Logging

from pdb import set_trace as bp  # This is only for debugging


#
#############################################################################
# Parent class for testing a model
#
class TesterBaseClass(object):
    def __init__(self, config, args):
        # Paths
        self.root_dir = config["paths"][
            "root_dir"
        ]  # directory of the repository
        
        # directory of the dataset
        self.data_dir = os.path.join(
            config["paths"]["data_prefix"],
            config["datasets"][args.dataset]["data_dir"]
            ) 

        use_case_dir = "privacy"

        self.model_dir = os.path.join(
            self.root_dir, 
            "trained_models",
            use_case_dir,
            args.dataset.lower()
            )  # directory where the model is saved
        
        self.res_dir =  os.path.join(
            self.root_dir, 
            "results",
            use_case_dir,
            args.dataset.lower()
            )  # directory where to save the predictions

        self.params = config["params"]
        self.num_workers = config["num_workers"]

        # Boolean for using binary cross-entropy loss
        self.b_bce = args.use_bce

        # --------------------------------------------
        # Model network
        self.n_out_classes = config["net_params"]["num_out_classes"]
        # self.num_obj_cat = config["net_params"]["num_obj_cat"]
        self.n_graph_nodes = config["net_params"]["n_graph_nodes"]
        self.node_feature_size = config["net_params"]["node_feat_size"]

        # --------------------------------------------
        self.net = None

        self.model_mode = args.model_mode

    def get_filename(self, model_name, extension=".csv", prefix="", suffix=""):
        """Create a filename based on the fold ID and model name.

        The function returns the filename with any additional prefix appended,
        and the extension based the argument passed (.csv as default).
        """
        if self.params["training_mode"] == "crossval":
            filename = "{:s}-{:d}".format(model_name, self.params["fold_id"])

        if self.params["training_mode"] == "final":
            filename = "{:s}-final".format(model_name)

        if self.params["training_mode"] == "original":
            filename = "{:s}-original".format(model_name)

        filename = prefix + filename + suffix + extension

        return filename

    def get_model(self, model_filename, model_name):
        # create .txt file to log results
        self.checkpoint_dir = os.path.join(
            self.model_dir,
            # "{:d}-class".format(self.n_out_classes),
            model_name,
        )

        fullpathname = os.path.join(self.checkpoint_dir, model_filename)
        checkpoint = torch.load(fullpathname)

        # self.s2p_net.load_state_dict(checkpoint["net"])
        return checkpoint

    def save_predictions(
        self, model_name, img_arr, prediction_scores, cm_pred, _suffix=""
    ):
        df_data = {
            "image": img_arr,
            "probability": prediction_scores,
            "pred_class": cm_pred,
        }
        df = pd.DataFrame(df_data)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir, exist_ok=True)

        filename = self.get_filename(model_name, extension=".csv", suffix=_suffix)

        df.to_csv(
            os.path.join(self.res_dir, filename),
            index=False,
        )

        print("Predictions saved in " + os.path.join(self.res_dir, filename))

    def load_testing_graph_data(
        self, 
        partition, 
        dataset_name, 
        net_params, 
        b_filter_imgs=False, 
        split_mode="test"
        ):
        """ """
        data_wrapper = WrapperDatasets(
            root_dir=self.root_dir,
            data_dir=self.data_dir,
            num_classes=self.n_out_classes,
            fold_id=self.params["fold_id"], 
            graph_mode=net_params["graph_type"],           
            # adj_mat_fn=self.adjacency_filename,
            # n_graph_nodes=self.n_graph_nodes,
            # node_feat_size=self.node_feature_size,
            # graph_mode=net_params["graph_type"],
            n_graph_nodes=net_params["n_graph_nodes"],
            node_feat_size=net_params["node_feat_size"],
        )

        data_wrapper.load_split_set(
            dataset_name,
            partition=partition,
            mode="train" if split_mode == "val" else split_mode,
            b_use_card=net_params["use_card"], 
            b_use_conf=net_params["use_conf"],
            b_filter_imgs=b_filter_imgs,
            )

        self.testing_loader = DataLoader(
            data_wrapper.get_data_split(split_mode),
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.data_wrapper = data_wrapper

    def test(self):
        """Test the model"""
        print("\nTesting ...")

        self.net = self.net.to(device)
        self.net.eval()

        # Initialise testing variables
        cm_pred = []
        prediction_scores = []
        sample_arr = []

        start_batch_time = time.time()

        with torch.no_grad():
            for batch_idx, (
                node_feats_var,
                target,
                weight,
                adj_mat,
                sample_name,
            ) in enumerate(tqdm(self.testing_loader, ascii=True)):
                # node_feats = torch.squeeze(node_feats, -1)
                # node_feats = node_feats.to(device)

                # # Run forward of the model (return logits)
                # outputs = self.net(node_feats)

                # Run forward of the model (return logits)
                outputs = self.net(node_feats_var.to(device), adj_mat)

                if self.b_bce:
                    assert self.n_out_classes == 2

                    assert len(outputs.shape) in [1, 2]

                    if len(outputs.shape) == 1:
                        out_logits = outputs
                    elif (len(outputs.shape) == 2) & (outputs.shape[1] == 1):
                        out_logits = outputs[:, 0]

                    # convert logits into probabilities
                    out_probs = torch.sigmoid(out_logits)

                    # Round the probabilities to determine the class
                    preds = out_probs.round()

                    # Convert predictions (tensor) into a list
                    preds = preds.data.cpu().numpy().tolist()
                    prediction_scores.append(preds)

                else:
                    assert self.n_out_classes >= 2

                    # Compute the predicted class for all data with softmax
                    outputs_np = F.softmax(outputs, dim=1).data.cpu().numpy()
                    # preds = list(np.argmax(outputs_np, axis=1))
                    preds = np.argmax(outputs_np, axis=1)
                    prediction_scores.append(outputs_np[:, 0])

                # Add the predictions in current batch to all predictions
                cm_pred = np.concatenate([cm_pred, preds])

                # Reset the batch time
                sample_arr = np.concatenate([sample_arr, list(sample_name)])
                # img_arr.append(image_name)

                start_batch_time = time.time()

        # Prepare data for saving
        sample_arr2 = [sample for sample in sample_arr]
        pred_scores_l = [
            num for sublist in prediction_scores for num in sublist
        ]

        self.save_predictions(
            self.net.get_model_name(), sample_arr2, pred_scores_l, cm_pred,
            _suffix="_" + self.model_mode
        )


    def test_rule(self):
        """Test the model"""
        print("\nTesting ...")

        # Initialise testing variables
        cm_pred = []
        prediction_scores = []
        sample_arr = []

        start_batch_time = time.time()

        with torch.no_grad():
            for batch_idx, (
                node_feats_var,
                target,
                weight,
                adj_mat,
                sample_name,
            ) in enumerate(tqdm(self.testing_loader, ascii=True)):

                # Run forward of the model (return logits)
                outputs = self.net(node_feats_var, adj_mat)

                # Compute the predicted class for all data with softmax
                outputs_np = F.softmax(outputs, dim=1).data.cpu().numpy()
                # preds = list(np.argmax(outputs_np, axis=1))
                preds = np.argmax(outputs_np, axis=1)
                prediction_scores.append(outputs_np[:, 0])

                # Add the predictions in current batch to all predictions
                cm_pred = np.concatenate([cm_pred, preds])

                # Reset the batch time
                sample_arr = np.concatenate([sample_arr, list(sample_name)])
                # img_arr.append(image_name)

                start_batch_time = time.time()

        # Prepare data for saving
        sample_arr2 = [sample for sample in sample_arr]
        pred_scores_l = [
            num for sublist in prediction_scores for num in sublist
        ]

        self.save_predictions(
            self.net.get_model_name(), sample_arr2, pred_scores_l, cm_pred,
            _suffix="_" + self.model_mode
        )

    def run(self):
        """ """
        filename = self.get_filename(
            self.net.get_model_name(), extension=".txt", suffix="_testing_log"
        )

        self.log = Logging()
        self.log.initialise(os.path.join(self.checkpoint_dir, filename))
        self.log.write_preamble(self.net.get_model_name(), self.n_out_classes)

        if self.net.get_model_name() in ["personrule"]:
            self.test_rule()
        else:
            self.test()

        self.log.write_ending()


#############################################################################
#
