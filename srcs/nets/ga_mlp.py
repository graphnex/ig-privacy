#!/usr/bin/env python
#
# Model and forward pass for the graph-agnostic baseline with MLPs.
#
# Partially taken from:
# https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/nets/superpixels_graph_classification/mlp_net.py
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/09/11
# Modified Date: 2023/09/11
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
# ----------------------------------------------------------------------------

import os
import sys
import argparse

# PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

np.set_printoptions(threshold=sys.maxsize, precision=2)

# PyTorch Geometric libraries
from torch_geometric.nn import global_mean_pool, global_add_pool

from srcs.nets.MLPReadout import MLPReadout

from srcs.utils import (
    device,
    print_model_parameters,
)

from pdb import set_trace as bp

#############################################################################

class GraphAgnosticMLP(nn.Module):
    """Graph-agnostic baseline with multi-layer perceptrons.

    Preparation and forward pass for the multi-layer perceptron.

    Simple graph-agnostic baseline that parallelly applies an MLP on each
    node’s feature vector, independent of other nodes. For graph-level
    classification, the node features are pooled together to form a global
    feature that is then passed to a classifer (3-layers MLP, or MLP Readout).

    Partially taken from https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/nets/superpixels_graph_classification/mlp_net.py
    """

    def __init__(
        self,
        config,
    ):
        """Constructor of the class"""
        super(GraphAgnosticMLP, self).__init__()

        self.model_name = config["model_name"]

        net_params = config["net_params"]

        self.node_feat_size = net_params["node_feat_size"]

        in_dim = net_params["node_feat_size"]
        hidden_dim = net_params["hidden_dim"]
        n_graph_nodes = net_params["n_graph_nodes"]
        n_classes = net_params["num_out_classes"]
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout = net_params["dropout"]
        n_layers = net_params["num_layers"]
        b_batch_norm = net_params["use_bn"]

        self.gated = net_params["gated"]
        self.readout_mode = net_params["readout"]

        self.max_num_roi = net_params["max_num_roi"]

        # b_class_nodes = net_params["use_class_nodes"]
        # 
        # n_obj_cat = net_params["num_obj_cat"]
        # n_scene_cat = net_params["num_scene_cat"]
        # 
        # if b_class_nodes:
        #     assert(n_graph_nodes == (n_scene_cat + n_obj_cat + n_classes))
        # else:
        #     assert(n_graph_nodes == (n_scene_cat + n_obj_cat))
        

        # self.dropout = nn.Dropout(p=dropout_prob)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.b_use_embedding = False
        
        self.output_proba = True if "output_proba" in config["net_params"] else False
        if self.output_proba:
            self.use_bce = config["params"]["use_bce"]

        if "use_embedding" in net_params:
            self.b_use_embedding = net_params["use_embedding"]

            if self.b_use_embedding:
                try:
                    self.embedding_size = net_params["embedding_size"]
                except:
                    raise ValueError(
                        "embdedding_size parameter missing in the configuration file!"
                    )

                if self.node_feat_size > 1:
                    # The next 2 embedding layers are independent for projecting
                    # cardinality and confidence separetely, when both are used
                    self.embedding1 = nn.Linear(
                        # net_params["node_feat_size"],
                        1,
                        self.embedding_size,
                        bias=True,
                    )
                    self.embedding2 = nn.Linear(
                        # net_params["node_feat_size"],
                        1,
                        self.embedding_size,
                        bias=True,
                    )

                    in_dim = self.embedding_size * self.node_feat_size

                else:
                    self.embedding = nn.Linear(
                        # net_params["node_feat_size"],
                        1,
                        self.embedding_size,
                        bias=True,
                    )

                    in_dim = self.embedding_size

        # Hidden layers
        feat_list = [in_dim] + [hidden_dim] * n_layers

        print("Feature list dimensions: ")
        print(feat_list)

        feat_mlp_modules = []

        for l_idx in range(n_layers):
            feat_mlp_modules.append(
                nn.Linear(feat_list[l_idx], feat_list[l_idx + 1], bias=True)
            )

            # Add batch normalisation layer (if activated)
            # - currently not working
            if b_batch_norm:
                feat_mlp_modules.append(nn.BatchNorm1d(n_graph_nodes))

            feat_mlp_modules.append(nn.ReLU())
            feat_mlp_modules.append(nn.Dropout(dropout))

            print("Layer")
            print_model_parameters(
                nn.Linear(feat_list[l_idx], feat_list[l_idx + 1], bias=True)
            )
            print_model_parameters(nn.BatchNorm1d(n_graph_nodes))

        self.feat_mlp = nn.Sequential(*feat_mlp_modules)

        if self.gated:
            self.gates = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Fully connected layer for output
        self.readout_mlp = MLPReadout(feat_list[-1], n_classes)

        self.print_number_parameters()

        print("GA-Seed: {:d}".format(torch.random.initial_seed()))

    def print_number_parameters(self):
        """ """
        print("\nEmbeddings: ")
        if self.node_feat_size == 2:
            print_model_parameters(self.embedding1)
            print_model_parameters(self.embedding2)

        elif self.node_feat_size == 1:
            print_model_parameters(self.embedding)

        print("\nDropout: ")
        print_model_parameters(self.in_feat_dropout)

        # Print classifier parameters to screen
        print("\nGraph-agnostic MLP parameters: ")
        print_model_parameters(self.feat_mlp)

        print("\nMLP Readout parameters: ")
        print_model_parameters(self.readout_mlp)

    def print_embedding_param(self):
        """
        """
        print("Parameters of embedding 1")
        
        print(torch.t(self.embedding1.weight.data))
        print(self.embedding1.bias.data)

    def get_model_name(self):
        return self.model_name

    def set_batch_size(self, batch_size):
        """Dummy function based on other networks"""
        batch_sz = batch_size

    def forward(self, x, adj_mat):
        """Forward pass.

        Arguments:
            - x: node features
            - adjmat:   the adjacency matrix of the corresponding graph. This
                        argument is present for consistency with other GNNs
                        implementations, but it is not used in this class.

        Return:
            - logits:
        """

        # Prepare batch tensor for pooling
        batch_sz, n_feats, _ = x.shape
        batch = (
            torch.Tensor(np.array(range(batch_sz)))
            .unsqueeze(1)
            .repeat(1, n_feats)
            .view(-1)
        )
        batch = batch.to(dtype=torch.int64).to(device)

        if self.b_use_embedding:
            if self.node_feat_size == 2:
                # embeddings = self.embedding(torch.unsqueeze(x.to(device),3))
                # embeddings = embeddings.view(x.shape[0],x.shape[1],-1)
                embeddings1 = self.embedding1(
                    torch.unsqueeze((x[:, :, 0]), 2)
                )
                embeddings2 = self.embedding2(
                    torch.unsqueeze((x[:, :, 1]), 2)
                )
                embeddings = torch.cat((embeddings1, embeddings2), 2)
            else:
                embeddings = self.embedding(x)
            x_d = self.in_feat_dropout(embeddings)  # Initial dropout
        else:
            x_d = self.in_feat_dropout(x.to(device))  # Initial dropout

        x_mlp = self.feat_mlp(x_d)  # Run through the individual MLPs

        # Pooling
        if self.gated:
            # Add sum of nodes (replace the DGL-based one)
            x_mlp = torch.sigmoid(self.gates(x_mlp)) * x_mlp
            x_mlp_global = global_add_pool(
                x_mlp.view(batch_sz * n_feats, -1), batch
            )
        else:
            if self.readout_mode == "mean":
                x_mlp_global = global_mean_pool(
                    x_mlp.view(batch_sz * n_feats, -1), batch
                )
            elif self.readout_mode == "sum":
                x_mlp_global = global_add_pool(
                    x_mlp.view(batch_sz * n_feats, -1), batch
                )

        # Final classifier (based on an MLP)
        logits = self.readout_mlp(x_mlp_global)

        # Optional: sigmoid or softmax function
        if self.output_proba:
            if self.use_bce:
                logits = logits[:, 0]
                return torch.sigmoid(logits)
            else:
                return F.softmax(logits, dim=1)
        else:
            # if self.b_softmax:  ## IS IT USED SOMEWHERE?
            #     probs = F.softmax(logits, dim=1)
            #     return probs
            # else:
            return logits

        return logits
