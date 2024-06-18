#!/usr/bin/env python
#
# MLP
#
##############################################################################
# Authors:
# - Myriam Bontonou, myriam.bontonou@ens-lyon.fr
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/05/19
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
# Libraries
import os
import sys
import argparse

# PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from srcs.utils import (
    device,
    print_model_parameters,
)

from pdb import set_trace as bp

#############################################################################
class MLP(nn.Module):
    """Multi-layer perceptron.

    Preparation and forward pass for the multi-layer perceptron.
    """

    def __init__(
        self,
        config,
    ):
        """Constructor of the class

        Arguments:
            - n_layers: number of layers
            - n_features: number of features
            - n_features_layer: number of features per layer
            - n_out_classes: number of output classes (2 for binary classification)
            - b_batch_norm: boolean to use the batch normalization. Deafult: false
            - b_softmax: boolean to use softmax. Default: false
            - dropout_prob: probability for the droput layer. Default: 0
        """
        super(MLP, self).__init__()

        self.model_name = config["model_name"]

		# Hyperparameters
        net_params = config["net_params"]

        self.node_feat_size = net_params["node_feat_size"]

        self.output_proba = True if "output_proba" in config["net_params"] else False
        if self.output_proba:
            self.use_bce = config["params"]["use_bce"]

        in_dim = net_params["n_graph_nodes"] * net_params["node_feat_size"]
        hidden_dim = net_params["hidden_dim"]
        n_classes = net_params["num_out_classes"]
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout_prob = net_params["dropout"]
        n_layers = net_params["num_layers"]
        b_batch_norm = net_params["use_bn"]

        self.dropout = nn.Dropout(p=dropout_prob)

        self.b_softmax = net_params["b_softmax"] ## IS IT REALLY USED?

        # Hidden layers
        feat_list = [in_dim] + [hidden_dim] * n_layers

        layers = []
        for l_idx in range(n_layers):
            layers.append(nn.Linear(feat_list[l_idx], feat_list[l_idx + 1]))

            # Add batch normalisation layer (if activated)
            if b_batch_norm:
                # layers.append(nn.BatchNorm1d(feat_list[l_idx + 1]))
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

        # Fully connected layer for output
        self.fc = nn.Linear(feat_list[-1], n_classes)

    def print_number_parameters(self):
        """ Print classifier parameters to screen."""
        print("\nMLP Layers parameters: ")
        print_model_parameters(self.layers)

        print("\nFinal fully connected layer parameters: ")
        print_model_parameters(self.fc)

    def get_model_name(self):
        return self.model_name

    def set_batch_size(self, batch_size):
        """Dummy function based on other networks"""
        batch_sz = batch_size

    def forward(self, x, adj_mat):
        """Forward pass.

        Arguments:
            - x: node features
            - adj_mat: adjacency matrix

        Return the output of the MLP before or after a sigmoid or a softmax function.
        """
        # Input data
        if self.node_feat_size == 1:
            x = torch.squeeze(x, -1)
        else:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        
        # This might not be needed
        x = x.to(device)

        # Layers
        x_d = self.dropout(x)
        x_l = self.layers(x_d)
        logits = self.fc(x_l)

        # Optional: sigmoid or softmax function
        if self.output_proba:
            if self.use_bce:
                logits = logits[:, 0]
                return torch.sigmoid(logits)
            else:
                return F.softmax(logits, dim=1)
        else:
            if self.b_softmax:  ## IS IT USED SOMEWHERE?
                probs = F.softmax(logits, dim=1)
                return probs
            else:
                return logits


