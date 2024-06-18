#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/08/30
# Modified Date: 2023/09/07
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
import inspect

# setting path
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.insert(0, pp_dir)

import numpy as np

np.set_printoptions(threshold=sys.maxsize, precision=2)

import pandas as pd
import json

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import Sequential, GCNConv, global_mean_pool

# Utilities
from srcs.nets.grm import GraphReasoningModel
from srcs.utils import (
    convert_adj_to_edge_index,
    convert_adj_to_edge_index_weight,
    check_if_square,
    check_if_symmetric,
    device,
    print_model_parameters,
)

from pdb import set_trace as bp


#############################################################################
#
class GraphPrivacyAdvisor(nn.Module):
    def __init__(self, config):
        super(GraphPrivacyAdvisor, self).__init__()

        self.model_name = config["model_name"]

        self.net_params = config["net_params"]

        self.node_feature_size = self.net_params["node_feat_size"]

        # Number of output classes (privacy levels)
        self.n_out_classes = self.net_params["num_out_classes"]

        # Number of object categories (for COCO=80, no background)
        self.n_obj_cats = self.net_params["num_obj_cat"]

        self.n_graph_nodes = self.n_obj_cats + self.n_out_classes

        self.graph_mode = self.net_params["graph_mode"]

        self.b_bce = config["params"]["use_bce"]

        self.b_bip_undirected = self.net_params["use_bip_undirected"]

        self.in_feat_dropout = nn.Dropout(self.net_params["in_feat_dropout"])

        in_feat_dim = self.node_feature_size
        self.b_use_embedding = False

        if "use_embedding" in self.net_params:
            self.b_use_embedding = self.net_params["use_embedding"]

            if self.b_use_embedding:
                try:
                    self.embedding_size = self.net_params["embedding_size"]
                except:
                    raise ValueError(
                        "embdedding_size parameter missing in the configuration file!"
                    )

                self.embedding = nn.Linear(
                    # self.node_feature_size,
                    1,
                    self.embedding_size,
                    bias=True,
                )

                if self.node_feature_size > 1:
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

                    in_feat_dim = self.embedding_size * self.node_feature_size
                else:
                    in_feat_dim = self.embedding_size

        if self.net_params["b_flag_type"]:
            if self.net_params["flag_mode"] == "flag":
                in_feat_dim +=1
            elif self.net_params["flag_mode"] == "one-hot":
                in_feat_dim +=2

        # Initialise the graph convolutional network
        self.gnn = GraphReasoningModel(
            # grm_hidden_channel=self.net_params["ggnn_hidden_channel"],
            grm_hidden_channel=in_feat_dim,
            grm_output_channel=self.net_params["ggnn_output_channel"],
            time_step=self.net_params["time_step"],
            n_out_class=self.net_params["num_out_classes"],
            n_obj_cls=self.net_params["num_obj_cat"],
            attention=self.net_params["use_attention"],
            b_bce=config["params"]["use_bce"],
        )

        print("\nGRM parameters: ")
        print_model_parameters(self.gnn)

        self.reshape_input = nn.Linear(
            (self.net_params["num_obj_cat"] + 1)
            * self.net_params["num_out_classes"],
            (self.net_params["num_obj_cat"] + 1),
        )

        self.initialise_classifier(config["params"]["use_bce"])

        # self.rng = np.random.default_rng()

    def get_model_name(self):
        return self.model_name

    # def initialise_classifier(self, use_bce):
    #     """ """

    #     if use_bce:
    #         assert self.n_out_classes == 2
    #         fc_out = nn.Linear(self.net_params["node_feat_size"], 1)

    #     else:
    #         assert self.n_out_classes >= 2
    #         # fc_out = nn.Linear(self.node_feature_size, self.n_out_classes)
    #         fc_out = nn.Linear(self.net_params["node_feat_size"], 1)

    #     self.classifier = nn.Sequential(
    #         nn.Dropout(),
    #         nn.Linear(
    #             (self.net_params["num_obj_cat"] + 1) * self.net_params["ggnn_output_channel"],
    #             self.net_params["node_feat_size"],
    #         ),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         fc_out,
    #     )

    #     # Initialise the weights of the final classifier using Xavier's uniform.
    #     for m in self.classifier.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight.data)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()

    #     # Print classifier parameters to screen
    #     print("\nClassifier parameters: ")
    #     print_model_parameters(self.classifier)

    def initialise_classifier(self, use_bce):
        """ """

        if use_bce:
            assert self.n_out_classes == 2
            fc_out = nn.Linear(self.net_params["num_out_classes"], 1)

        else:
            assert self.n_out_classes >= 2
            # fc_out = nn.Linear(self.node_feature_size, self.n_out_classes)
            fc_out = nn.Linear(self.net_params["num_out_classes"], 1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                self.net_params["num_obj_cat"] + 1,
                self.net_params["num_out_classes"],
            ),
            nn.ReLU(True),
            nn.Dropout(),
            fc_out,
        )

        # Initialise the weights of the final classifier using Xavier's uniform.
        for m in self.classifier.modules():
            cnt = 0
            if isinstance(m, nn.Linear):
                if cnt == 0:
                    m.weight.data.normal_(0, 0.001)
                else:
                    m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                cnt += 1

        # Print classifier parameters to screen
        print("\nClassifier parameters: ")
        print_model_parameters(self.classifier)

    def initialise_prior_graph(self, prior_graph):
        # This is to avoid computing multiple times
        self.prior_graph = prior_graph
        (
            self.adj_mat_obj_occ,
            self.adj_mat_bipartite,
        ) = self.prior_graph.split_adjacency_matrix()

        if self.b_bip_undirected:
            if not check_if_symmetric(self.adj_mat_bipartite):
                self.adj_mat_bipartite += self.adj_mat_bipartite.transpose()

                assert check_if_symmetric(self.adj_mat_bipartite)

        if self.graph_mode == "bipartite":
            # if self.b_bip_undirected:
                # TODO

            self.gnn.set_adjacency_matrix(self.adj_mat_bipartite, self.net_params["mode_adj"])
        else:
            # Binarise the adjacency matrix (as defined in GPA paper)
            self.adj_mat_obj_occ[self.adj_mat_obj_occ > 0] = 1

            self.gnn.set_adjacency_matrix(self.adj_mat_obj_occ, self.net_params["mode_adj"])

    def set_batch_size(self, batch_size):
        """Dummy function based on other networks"""
        # batch_sz = batch_size
        self.gnn.set_batch_size(batch_size)


    def add_flag(self, node_feats, mode="flag"):
        """
        
        From Content-based Graph Privacy Advisor:
        "f_h is a one-hot vector that differentiates between scene and object 
        information and ⊕ is the concatenation operation."
        Source: https://arxiv.org/pdf/2210.11169.pdf

        The implementation is represented as a simple flag (Boolean variable)
        that is set to 1 only for object categories, and 0 for the two privacy
        nodes. This is not a one-hot vector. In our implementation, we include
        both cases.

        Mode:
            - "flag":
            - "one-hot": 

        Potential issues (not verified): 
            * increased computational time due to repetition of the operation,
            as well as the moving to GPU, dependency on the input batch size.
        """
        num_imgs, n_feats, _ = node_feats.shape

        if mode == "flag":
            feat_type = torch.zeros(num_imgs, self.n_graph_nodes,1)
            feat_type[:,self.n_out_classes:] = 1

        elif mode == "one-hot":
            feat_type = torch.zeros(num_imgs, self.n_graph_nodes,2)
            feat_type[:, :self.n_out_classes  ,0] = 1
            feat_type[:,  self.n_out_classes: ,1] = 1

        return torch.cat((feat_type.to(device),node_feats), 2)

    def set_rand_privacy_feats(self, node_feats):
        """
        """
        _, _, n_feats = node_feats.shape

        r1 = -20
        r2 = 20
        
        node_feats[:, :self.n_out_classes, :] = torch.FloatTensor(n_feats).uniform_(r1, r2)

        return node_feats

    def forward(self, node_features, adj_mat):
        """
        Forward pass through the model (cannot be interrupted).

        Note that adj_mat is not used in this model. Adj_mat is the adjacency 
        matrix for each sample data, whereas in this code we use the prior
        graph. 
        """
        num_imgs, n_feats, _ = node_features.shape
        
        if self.net_params["rand_priv"]:
            node_features = self.set_rand_privacy_feats(node_features)

        if self.b_use_embedding:
            if self.node_feature_size == 2:
                # embeddings = self.embedding(torch.unsqueeze(x.to(device),3))
                # embeddings = embeddings.view(x.shape[0],x.shape[1],-1)
                embeddings1 = self.embedding1(
                    torch.unsqueeze((node_features[:, :, 0]), 2)
                )
                embeddings2 = self.embedding2(
                    torch.unsqueeze((node_features[:, :, 1]), 2)
                )
                embeddings = torch.cat((embeddings1, embeddings2), 1)
            else:
                embeddings = self.embedding(node_features)
            x_d = self.in_feat_dropout(embeddings)
        else:
            x_d = node_features

        if self.net_params["b_flag_type"]:
            x_d = self.add_flag(x_d, mode=self.net_params["flag_mode"])

        model_input = x_d.view(num_imgs, -1)
        
        grm_feature = self.gnn(model_input)

        # if self.mode in [a for a in range(0,6)]:
        grm_feature = self.reshape_input(grm_feature)

        nodes_unnormalized_scores = self.classifier(grm_feature).view(
            num_imgs, -1
        )

        if self.b_bce:
            nodes_unnormalized_scores = nodes_unnormalized_scores.squeeze()

        nodes_unnormalized_scores.float()

        return nodes_unnormalized_scores
