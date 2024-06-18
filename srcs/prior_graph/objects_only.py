#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2024/02/20
# Modified Date: 2024/02/20
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

# setting path
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.insert(0, pp_dir)

import argparse  # Parser for command-line options, arguments and sub-commands

import pandas as pd  # Open-source data analysis and manipulation tool
import numpy as np  # Scientific computing

import itertools

from tqdm import tqdm  # smart progress meter for loops

import json  # Open standard file format to store data

import networkx as nx  # Software for complex networks (including graphs)
import matplotlib.pyplot as plt  # Visualisation tool

from srcs.prior_graph.prior_graph_base import PriorKnowlegeGraphBase

from pdb import set_trace as bp  # For debugging and adding breakpoints


#############################################################################
class PriorGraphObjectsOnly(PriorKnowlegeGraphBase):
    def __init__(self, config):
        super().__init__(config)

        self.dirname = "obj-only"

        self.config = config # store config for later uses

        self.partition = config["params"]["training_mode"]

        self.num_obj_cat = config["net_params"]["num_obj_cat"]
        
        try:
            if config["net_params"]["num_scene_cat"] is not None:
                assert(config["net_params"]["num_scene_cat"] == 0)
        except:
            print("Field num_scene_cat not present! Continue with 0 scene categories.")
        
        self.num_scene_cat = 0

        self.b_self_edges = config["net_params"]["self_loop"]
    
        self.b_special_nodes = config["net_params"]["use_class_nodes"]

        if self.b_special_nodes:
            self.n_class_nodes = config["net_params"]["num_out_classes"]
            self.n_graph_nodes = (
                self.num_obj_cat + self.num_scene_cat + self.n_class_nodes
            )

            self.b_directed = True  # Only for special nodes

            self.n_out_classes = self.n_class_nodes
        else:
            self.n_graph_nodes = self.num_obj_cat + self.num_scene_cat

            self.b_directed = False

        self.b_weighted = True

        self.category = config["category_mode"]

        self.set_nodes_info()

    def add_edge_weight(self, idx1, idx2, prior_graph, weights, symm=False):
        """ """
        if idx1 in prior_graph:
            if idx2 not in prior_graph[idx1]:
                prior_graph[int(idx1)].append(idx2)
                weights[int(idx1)][int(idx2)] = 1

                if symm:
                    self.n_edges_added += 1
            else:
                weights[int(idx1)][int(idx2)] += 1
        else:
            prior_graph[int(idx1)] = [idx2]
            weights[int(idx1)][int(idx2)] = 1

            if symm:
                self.n_edges_added += 1

        return prior_graph, weights

    def normalise_prior_graph_weights(self, weights, cats_occ):
        """ """
        for n in range(self.n_graph_nodes):
            for key, val in weights[n].items():
                if self.b_special_nodes and key in range(self.n_out_classes):
                    weights[n][key] = val / cats_occ[key]
                    # weights[n][key] = val / cats_occ[n]
                else:
                    weights[n][key] = cats_occ[n] and val / cats_occ[n] or 0

        return weights

    def compute_weighted_graph_from_file(self):
        """
        Compute the weighted graph of objects and special nodes from the training set.

        For each image in the training split of a given dataset, the co-occurence
        of object categories identified in the
        """

        prior_graph = dict()
        weights = dict()

        for n in range(self.n_graph_nodes):
            prior_graph[n] = []
            weights[n] = dict()

        # Store the number of occurences of each category across the whole
        # training set
        cats_occ = np.zeros(self.n_graph_nodes)

        for img_idx in tqdm(range(self.n_imgs)):
            if self.dataset == "IPD":
                fullpath = os.path.join(
                    self.data_prefix[img_idx],
                    "graph_data",
                    self.dirname,
                    "node_feats",
                    self.l_imgs[img_idx].split(".")[0] + ".json",
                )
            else:
                fullpath = os.path.join(
                    self.data_dir,
                    "graph_data",
                    self.dirname,
                    "node_feats",
                    self.l_imgs[img_idx].split(".")[0] + ".json",
                )

            retval = False

            node_feats = json.load(open(fullpath))
            node_cat = []

            for node in node_feats:
                n_id = node["node_id"]

                # This is to add to the list only the concept nodes (objects and scenes)
                # whose cardinality is greater than 0
                # (or simply to exclude the privacy nodes in the format of the node features)
                if node["node_feature"][0] > 0:
                    node_cat.append(n_id)

                    if not retval:
                        retval = True

            # Continue if the image does not contain any concept
            if not retval:
                continue

            # Shift the categories by the number of special nodes representing
            # the privacy classes
            img_cats = np.array(node_cat)
            img_cats_unique = np.unique(img_cats)

            all_edges_dir = list(
                itertools.combinations(np.sort(img_cats_unique), 2)
            )

            # Count the occurrence of the number of categories and
            # add self-edges if the corresponding Boolean parameter was set
            for idx in img_cats_unique:
                cats_occ[idx] += 1

                if self.b_self_edges:
                    if img_cats[img_cats == idx].shape[0] > 1:
                        all_edges_dir.append((idx, idx))

            for edge in all_edges_dir:
                prior_graph, weights = self.add_edge_weight(
                    edge[0], edge[1], prior_graph, weights
                )

                # Other direction (symmetric)
                prior_graph, weights = self.add_edge_weight(
                    edge[1], edge[0], prior_graph, weights, True
                )

            if self.b_special_nodes:
                for idx in img_cats_unique:
                    for sp_node_id in range(self.n_out_classes):

                        # Link image nodes to privacy class based on label of the image
                        if sp_node_id == self.l_labels[img_idx]:
                            prior_graph, weights = self.add_edge_weight(
                                idx, sp_node_id, prior_graph, weights
                            )

                            if not self.b_directed:
                                # prior_graph, weights = self.add_edge_weight(
                                #     idx, sp_node_id, prior_graph, weights, True
                                # )
                                # New corrected version
                                prior_graph, weights = self.add_edge_weight(
                                    sp_node_id, idx, prior_graph, weights, True
                                )

        if self.b_special_nodes:
            for sp_node_id in range(self.n_out_classes):
                cats_occ[sp_node_id] = (
                    np.array(self.l_labels) == sp_node_id
                ).sum()

        weights = self.normalise_prior_graph_weights(weights, cats_occ)

        # self.graph_d = dict(sorted(prior_graph.items()))
        self.graph_d = dict(sorted(weights.items()))

    def save_weighted_prior_graph(self):
        """
        Save the weighted prior graph to file in CSV format.

        The weighted prior graph is automatically saved in the pre-defined
        directory ``/workdir/assets/adjacency_matrix/<model_name>/``, where
        <model_name> is the name of the model (e.g., gpa). The file is saved
        under the name prior_graph_weighted_fX_cY.csv, where X is the fold ID
        and Y is the number of classes, when the prior graph is computed from
        all the images in the training set. Otherwise, the file is saved under
        the name name prior_graph_weighted_fX_cY_clsZ.csv, where Z is the
        output class number (e.g., 0 for private and 1 for public).

        The file is saved as a list of edges with their corresponding weights
        for each row. The columns of the CSV file are:
        - Node-1: the ID of the source node (float).
        - Node-2: the ID of the target node (float).
        - Weight: weight of the edge (float).
        """
        if self.partition == "final":
            fold_str = self.partition
        elif self.partition == "original":
            fold_str = self.partition
        else:
            fold_str = "f{:d}".format(self.fold_id)

        if self.category == -1:
            graph_fn = "prior_graph_weighted_{:s}_c{:d}.csv".format(
                fold_str, self.n_out_classes
            )
        else:
            graph_fn = "prior_graph_weighted_{:s}_c{:d}_cls{:d}.csv".format(
                fold_str, self.n_out_classes, self.category
            )

        fullpath = os.path.join(
            self.data_dir,
            "graph_data",
            self.dirname,
            "adj_mat",
            graph_fn,
        )

        dirname = os.path.dirname(fullpath)
        print(dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        edge_weight = np.array([])
        for n1, val in self.graph_d.items():
            for n2, w in sorted(val.items()):
                row = np.array([n1, n2, w])

                if edge_weight.size == 0:
                    edge_weight = np.hstack((edge_weight, row))
                else:
                    edge_weight = np.vstack((edge_weight, row))

        headers = ["Node-1", "Node-2", "Weight"]
        pd.DataFrame(edge_weight).to_csv(fullpath, header=headers, index=None)

    def run_compute_graph(self):
        """ """

        self.get_dataset_img_labels(self.config, "train")

        if self.b_weighted:
            self.compute_weighted_graph_from_file()
            self.save_weighted_prior_graph()
        else:
            self.get_graph_edges_from_files()
            self.save_graph_as_json()

        self.print_graph_stats()
