#!/usr/bin/env python
#
# Parent/base class for explaining a PyTorch model using captum librairy.
#
##############################################################################
# Authors:
# - Myriam Bontonou
#
#  Created Date: 2023/06/05
# Modified Date: 2023/10/27
#
# MIT License

# Copyright (c) 2023 GraphNEx

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
# Librairies
import os
import sys
from datetime import datetime
import inspect
import numpy as np
np.set_printoptions(threshold=sys.maxsize, precision=2)
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from captum.attr import IntegratedGradients

# Path
# sys.path.append(os.getcwd())
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.insert(0, pp_dir)

# G-architecture modules
from srcs.datasets.wrapper import WrapperDatasets
from srcs.logging_gnex import Logging
from srcs.utils import set_seed, device



# Parent class for explaining a model
class ExplainerBaseClass:
    def __init__(self, config, args):
        # Paths
        self.root_dir =  config["paths"]["root_dir"]  # directory of the repository
        self.data_dir = os.path.join(config["paths"]["data_prefix"], config["datasets"][args.dataset]["data_dir"]) # directory of the dataset
        if args.dataset in ["BRCA"]:
            use_case_dir = "biology"
        else:
            use_case_dir = "privacy"
        self.model_dir = os.path.join(
            self.root_dir, 
            "trained_models",
            use_case_dir,
            args.dataset.lower()
            )  # directory where the model is saved
        ## self.model_filename = args.model_filename  # name of the saved model
        self.res_dir = os.path.join(
            self.root_dir, 
            "results",
            use_case_dir,
            args.dataset.lower()
            )  # directory where to save the predictions

        # Hyperameters
        self.params = config["params"]
        self.num_workers = config["num_workers"]
        self.b_bce = config["params"]["use_bce"]  ## args.use_bce WHY USING ARGS HERE AND NOT CONFIG?

        # Data loader
        self.dataset = config["dataset"]
        self.split_mode = config["params"]["split_mode"]
        ##self.partition = args.partition
        ##self.batch_size = args.batch_size
        ##self.num_workers = args.n_workers

        # Model network
        self.model_name = config["model_name"]
        self.net_params = config["net_params"]
        self.n_out_classes = config["net_params"]["num_out_classes"]
        self.n_graph_nodes = config["net_params"]["n_graph_nodes"]
        self.node_feature_size = config["net_params"]["node_feat_size"]
        self.net = None
        self.model_mode = args.model_mode
        
        # XAI method
        self.XAI_params = config["XAI_params"]
        ##self.method = args.method  # "Integrated_Gradients" or "Kernel_Shap"
        ##self.n_steps = args.n_steps  # Used if method == "Integrated_Gradients". Integer.
        ##self.n_samples = args.n_samples  # Used if method == "Kernel_Shap". Integer.
        ##self.studied_class = args.studied_class  # Explained examples belong to a class listed in`studied_class`.


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
        self.checkpoint_dir = os.path.join(self.model_dir, model_name)
        fullpathname = os.path.join(self.checkpoint_dir, model_filename)
        checkpoint = torch.load(fullpathname)
        return checkpoint

    
    def load_data(self, dataset_name, split_mode="test", b_filter_imgs=False):  ## split is "train" or "test"
        data_wrapper = WrapperDatasets(
            root_dir=self.root_dir,
            data_dir = self.data_dir,
            fold_id = self.params["fold_id"],
            num_classes = self.n_out_classes,
            graph_mode = self.net_params["graph_type"],
            # adj_mat_fn = self.adjacency_filename,
            n_graph_nodes = self.n_graph_nodes,
            node_feat_size = self.node_feature_size,
        )
        data_wrapper.load_split_set(
            dataset_name,
            self.params["training_mode"],
            mode = "train" if split_mode == "val" else split_mode,
            b_use_card = self.net_params["use_card"], 
            b_use_conf = self.net_params["use_conf"],
            b_filter_imgs = b_filter_imgs,
            )

        loader = DataLoader(
            data_wrapper.get_data_split(split_mode),
            batch_size = self.params["batch_size"],
            shuffle = False,
            num_workers = self.num_workers,
        )

        # self.data_wrapper = data_wrapper
        return loader

    
    def load_baseline(self,):
        """Load the input example used as a reference. It is a tensor of shape (1, n_feat).
        """
        print(" ")
        print("Loading of the input used as a reference to contrast explanations.")
        baseline_type = self.XAI_params["baseline_type"]
        base_class = self.XAI_params["base_class"]
        if baseline_type == "average":
            base_class = self.XAI_params["base_class"]
            training_loader = self.load_data(self.dataset, "train")
            self.baseline = torch.zeros(1, self.n_graph_nodes, 1).to(device)
            count = 0
            for x, target, _, adj_mat, _ in training_loader:
                x = x[target == base_class]
                x = torch.squeeze(x, -1)  ## transform
                self.baseline += torch.sum(x, axis=0).reshape(1, -1, 1).to(device)
                count += x.shape[0]
            self.baseline = self.baseline / count
        else:
            loader = self.load_data(self.dataset, "train")
            for x, target, _, adj_mat, _ in loader:
                pass
            self.baseline = torch.zeros(1, self.n_graph_nodes, self.node_feature_size).to(device)
        # print(len(adj_mat))
        # if len(adj_mat) != 0:
        #    print(adj_mat.shape)
        if adj_mat != []:
            adj_mat = torch.unsqueeze(adj_mat[0], dim=0)  ## assuming the same graph is used for every example
            # print(adj_mat.shape)
        self.baseline_pred = self.net(self.baseline, adj_mat)
        self.baseline_pred = self.baseline_pred.detach().cpu().numpy()
        print(f"The output of the baseline is {self.baseline_pred}")


    def load_XAI_method(self,):
        if self.XAI_params["method"] == "Integrated_Gradients":
            xai = IntegratedGradients(self.net)
        elif self.XAI_params["method"] == "Kernel_Shap":
            xai = KernelShap(self.net)
        return xai


    def compute_attributions_from_data_tensors(self, xai, x, target, adj_mat):
        # print("x, target, baseline", x.shape, target.shape, self.baseline.shape)
        # print(adj_mat)
        if self.XAI_params["method"] == "Integrated_Gradients":
            # Parameter
            n_steps = self.XAI_params["n_steps"]
            # Set a placeholder for the adjacency matrix if it is not used by the model.
            if adj_mat == []:
                adj_mat = torch.zeros(x.shape[0], 1)
            # Estimate the attributions with a maximal error of 0.01.
            valid = False
            while not valid:
                attrs, gap = xai.attribute(x, target=target, n_steps=n_steps, baselines=self.baseline, additional_forward_args=adj_mat, internal_batch_size=self.params["batch_size"], return_convergence_delta=True)
                if torch.max(torch.abs(gap)).item() > 0.01:
                    n_steps += self.XAI_params["n_steps"]
                    print(f"Maximal gap: {np.round(torch.max(torch.abs(gap)).detach().cpu(), 6)}. There is at least one sample for which the difference between the sum of the attributions and the predictions is higher than 0.01. Run again IG with {n_steps} steps.")
                else:
                    valid = True
        elif self.XAI_params["method"] == "Kernel_Shap":
            n_samples = self.XAI_params["n_samples"]
            attrs = xai.attribute(x, target=target, n_samples=n_samples)
        return attrs


    def explain(self, split_mode):
        """Return importance scores attributed by an XAI self.method to each feature of each input.
        """
        print(" ")
        print(f"Explain the decisions of {self.model_name} on {split_mode} data.")
        # Studied classes
        studied_class = self.XAI_params["studied_class"]
        assert len(studied_class) > 0, "Provide a list of classes to consider for computing the attributions."

        # XAI method
        xai = self.load_XAI_method()

        # Dataloader
        data_loader = self.load_data(self.dataset, split_mode)
        n_sample = 0
        for x, target, _, _, _ in data_loader:
            n_sample += torch.sum(sum(target == c for c in studied_class)).item()

        # Compute the attributions
        attr = torch.zeros(n_sample, self.n_graph_nodes, self.node_feature_size).to(device)
        data = np.zeros((n_sample, x.shape[1], x.shape[2]))
        y_pred = np.ones(n_sample)
        y_true = np.ones(n_sample, dtype='int')
        data_name = []
        torch.manual_seed(1)
        count = 0
        for i, (x, target, _, adj_mat, name) in enumerate(data_loader):
            print(i, end='\r')
            # Data
            x = x[sum(target == c for c in studied_class).bool()]
            name = list(np.array(name)[sum(target == c for c in studied_class).bool()])
            # x = torch.squeeze(x, -1)
            target = target[sum(target == c for c in studied_class).bool()]
            batch_size = x.shape[0]
            x = x.to(device)
            target = target.to(device)
            # Attributions
            attrs = self.compute_attributions_from_data_tensors(xai, x, target, adj_mat)
            attr[count:count + batch_size, :] = attrs
            # Predictions
            outputs = self.net(x, adj_mat)
            _, pred = torch.max(outputs.data, 1)
            y_true[count:count + batch_size] = target.cpu().detach().numpy()
            y_pred[count:count + batch_size] = pred.cpu().detach().numpy()
            # Original data
            data[count:count + batch_size] = x.cpu().detach().numpy()
            data_name += name

            count = count + batch_size
        attr = attr.detach().cpu().numpy()

        # Save
        self.save_attributions(attr, data, y_pred, y_true, self.baseline.detach().cpu().numpy(), self.baseline_pred, split_mode, data_name)


    def save_attributions(self, features_score, data, predictions, true_labels, baseline, baseline_pred, split_mode, data_name):
        filename = self.get_filename(self.net.get_model_name(), extension=".npy", suffix=f"_{self.XAI_params['method']}_{split_mode}")
        np.save(os.path.join(self.res_dir, filename),
                {'features_score': features_score,
                 'data': data,
                 'predictions': predictions,
                 'true_labels': true_labels,
                 'baseline': baseline,
                 'baseline_pred': baseline_pred,
                 'data_name': data_name,
                })
      

    def load_attributions(self, split_mode):
        filename = self.get_filename(self.net.get_model_name(), extension=".npy", suffix=f"_{self.XAI_params['method']}_{split_mode}")
        checkpoint = np.load(os.path.join(self.res_dir, filename), allow_pickle=True).item()
        return checkpoint['features_score'], checkpoint['data'], checkpoint['predictions'], checkpoint['true_labels'], checkpoint['baseline'], checkpoint['baseline_pred'], checkpoint['data_name']


    def run(self):
        filename = self.get_filename(self.net.get_model_name(), extension=".txt", suffix="_explainability_log")
        self.log = Logging()
        self.log.initialise(os.path.join(self.checkpoint_dir, filename))
        self.log.log.write("Explainability method: " + self.XAI_params["method"] + "\n")
        self.load_baseline()
        self.explain(self.split_mode)
        self.log.write_ending()


