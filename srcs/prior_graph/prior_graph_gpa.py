#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/09/05
# Modified Date: 2023/09/05
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


##############################################################################
class PriorKnowlegeGraphGPA(PriorKnowlegeGraphBase):
    def __init__(self, config):
        super().__init__(config)

        # print(self.model_name)
        # print("Graph Privacy Advisor (GPA)")
        self.dirname = ""

        self.num_obj_cat = config["net_params"]["num_obj_cat"]
        self.n_out_classes = config["net_params"]["num_out_classes"]

        self.n_graph_nodes = self.num_obj_cat + self.n_out_classes

        self.config = config

        self.b_special_nodes = config["net_params"]["use_class_nodes"]
        if self.b_special_nodes:
            self.n_class_nodes = config["net_params"]["num_out_classes"]
            self.n_graph_nodes = self.num_obj_cat + self.n_class_nodes

            self.b_directed = True  # Only for special nodes

            self.n_out_classes = self.n_class_nodes
        else:
            self.n_graph_nodes = self.num_obj_cat

            self.b_directed = False

        self.category = config["category_mode"]

        self.b_weighted = True

        self.set_nodes_info()

    def get_image_objects(self, fullpath):
        """ """
        try:
            objs = json.load(open(fullpath))
        except:
            # print("Missing object image: {:s}".format(fullpath))
            # missing_object_img.append(img_name)
            return False, None

        if len(objs["categories"]) == 0:
            # print("Image with no detected objects: {:s}".format(fullpath))
            # missing_object_img.append(img_name)
            return False, None

        return True, objs

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
                    weights[n][key] = val / cats_occ[n]

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
                    "dets",
                    self.l_imgs[img_idx].split(".")[0] + ".json",
                )
            else:
                fullpath = os.path.join(
                    self.data_dir,
                    "dets",
                    self.l_imgs[img_idx].split(".")[0] + ".json",
                )

            retval, objs = self.get_image_objects(fullpath)

            if not retval:
                continue

            # Shift the categories by the number of special nodes representing
            # the privacy classes
            img_cats = np.array(objs["categories"]) + self.n_out_classes
            img_cats_unique = np.unique(img_cats)

            all_edges_dir = list(
                itertools.combinations(np.sort(img_cats_unique), 2)
            )

            if self.b_self_edges:
                for idx in img_cats_unique:
                    cats_occ[idx] += 1

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
                        if sp_node_id == self.l_labels[img_idx]:
                            prior_graph, weights = self.add_edge_weight(
                                idx, sp_node_id, prior_graph, weights
                            )

                        if not self.b_directed:
                            prior_graph, weights = self.add_edge_weight(
                                idx, sp_node_id, prior_graph, weights, True
                            )

        if self.b_special_nodes:
            for sp_node_id in range(self.n_out_classes):
                cats_occ[sp_node_id] = (
                    np.array(self.l_labels) == sp_node_id
                ).sum()

        weights = self.normalise_prior_graph_weights(weights, cats_occ)

        # self.graph_d = dict(sorted(prior_graph.items()))
        self.graph_d = dict(sorted(weights.items()))

    def get_graph_edges_from_files(self):
        """
        Compute the edges of the graphs with only COCO object categories as nodes.
        Detected objects are retrieved from the input file.

        The prior graph is computed as an adjacency list, i.e., for each node id,
        we store a list of node id connected to the node under consideration.
        For compacteness of storage and given the symmetric form of the
        undirected graph, only one direction is saved. This means that if exists an
        edge (2,3), we only store the following list 2: [3, ...] and not the viceversa
        3: [2, ...].
        """

        prior_graph = dict()
        for n in range(self.n_graph_nodes):
            prior_graph[n] = []

        # missing_object_img = []

        for img_idx in tqdm(range(self.n_imgs)):
            retval, objs = self.get_image_objects(self.l_imgs[img_idx])

            if not retval:
                continue

            # Shift the categories by the number of special nodes representing
            # the privacy classes
            img_cats = np.array(objs["categories"]) + self.n_out_classes
            img_cats_unique = np.unique(img_cats)

            all_edges_dir = list(
                itertools.combinations(np.sort(img_cats_unique), 2)
            )

            if self.b_self_edges:
                for idx in img_cats_unique:
                    if img_cats[img_cats == idx].shape[0] > 1:
                        all_edges_dir.append((idx, idx))

            for edge in all_edges_dir:
                if edge[0] in prior_graph:
                    if edge[1] not in prior_graph[edge[0]]:
                        prior_graph[int(edge[0])].append(edge[1])
                        self.n_edges_added += 1
                else:
                    prior_graph[int(edge[0])] = [edge[1]]
                    self.n_edges_added += 1

        self.graph_d = dict(sorted(prior_graph.items()))

        # print(
        #     "Number of missing object images: {:d}".format(
        #         len(missing_object_img)
        #     )
        # )

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

        if self.category == -1:
            graph_fn = "prior_graph_weighted_f{:d}_c{:d}.csv".format(
                self.fold_id, self.n_out_classes
            )
        else:
            graph_fn = "prior_graph_weighted_f{:d}_c{:d}_cls{:d}.csv".format(
                self.fold_id, self.n_out_classes, self.category
            )

        fullpath = os.path.join(
            self.data_dir,
            "graph_data",
            "{:d}-class".format(self.n_out_classes),
            "gpa",
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

        if self.b_weighted:
            self.compute_weighted_graph_from_file()
            self.save_weighted_prior_graph()
        else:
            self.get_graph_edges_from_files()
            self.save_graph_as_json()

        self.print_graph_stats()

    def run_graph_analysis(self):
        self.load_graph_nx()
        self.print_graph_stats()
        # self.save_graph_node_degrees()

    def load_adjacency_matrix_csv(self, filename):
        """
        The adjacency matrix of the 80 object categories from COCO
        from a specific training set was saved into NPZ format (numpy file)
        """
        print("Loading adjacency matrix (CSV) ...")

        df = pd.read_csv(filename, sep=",")

        ext_adj_mat = np.zeros((self.n_graph_nodes, self.n_graph_nodes))

        num_edges = df.shape[0]
        for j_iter in range(num_edges):
            idx1 = int(df.loc[j_iter, :][0]) + self.n_out_classes
            idx2 = int(df.loc[j_iter, :][1]) + self.n_out_classes
            value = df.loc[j_iter, :][2]

            ext_adj_mat[idx1, idx2] = value
            ext_adj_mat[idx2, idx1] = value

        self.adjacency_matrix = ext_adj_mat.astype(float)

    def load_adjacency_matrix_json(self, json_file_path):
        """
        The adjacency matrix of the 80 object categories from COCO
        from a specific training set was saved into JSON format
        """
        print("Loading adjacency matrix (JSON) ...")

        adj_l = json.load(open(json_file_path, "r"))

        adj_mat = np.zeros((self.n_graph_nodes, self.n_graph_nodes))

        for k, neighbours in adj_l.items():
            for v in neighbours:
                adj_mat[int(k), v] = 1.0
                adj_mat[v, int(k)] = 1.0

        return adj_mat.astype(float)

    def load_weighted_graph(self, filename):
        """-"""
        df = pd.read_csv(filename, sep=",")

        n_nodes = max(df["Node-1"].max(), df["Node-2"].max()) + 1
        assert self.n_graph_nodes == n_nodes

        edge_weights = df.to_numpy()

        adj_mat = np.zeros((self.n_graph_nodes, self.n_graph_nodes))

        n_rows, n_cols = edge_weights.shape

        for n in range(n_rows):
            idx1 = edge_weights[n][0]
            idx2 = edge_weights[n][1]
            w = edge_weights[n][2]
            adj_mat[int(idx1), int(idx2)] = w

        return adj_mat

        # def get_adjacency_matrix_filename(self, dataset, model_name):
        """ """
        # assert dataset in ["PicAlert", "VISPR", "PrivacyAlert", "IPD"]

        # if dataset == "PicAlert":
        #     dataset_name = "picalert"
        # elif dataset == "VISPR":
        #     dataset_name = "vispr"
        # elif dataset == "PrivacyAlert":
        #     dataset_name = "privacyalert"
        # elif dataset == "IPD":
        #     dataset_name = "ipd"

        # dataset_name = "ipd" # TO BE REMOVED

    # def load_adjacency_matrix(
    #     self,
    #     data_dir,
    #     model_name,
    #     partition,
    #     fold_id,
    #     mode="square_sym",
    #     self_edges=False,
    # ):
    #     """

    #     Arguments:
    #         - filename: name of the file where the adjacency matrix is stored and to load.
    #         - mode:
    #         - self_edges: boolean to include also self-edges in the graph.

    #     mode:
    #         - square_sym: a square and symmetric matrix representing an undirected graph
    #     """
    #     print()  # Empty line on the command line

    #     assert mode in [
    #         "square_sym",
    #         "ac",
    #     ]

    #     adj_mat_fullpath = os.path.join(
    #         data_dir,
    #         "graph_data",
    #         "{:d}-class".format(self.n_out_classes),
    #         model_name,
    #         "adj_mat",
    #     )

    #     if partition == "crossval":
    #         adj_mat_fn = "prior_graph_fold{:d}".format(fold_id)

    #     if partition == "final":
    #         adj_mat_fn = "prior_graph_final"

    #     if partition == "original":
    #         adj_mat_fn = "prior_graph_original"
    #         # adj_mat_fn = "prior_graph_fold{:d}".format(fold_id)

    #     adjacency_filename = os.path.join(adj_mat_fullpath, adj_mat_fn)

    #     if mode == "ac":
    #         adjacency_filename += "_ac.csv"
    #         # print(adjacency_filename)

    #         self.adjacency_matrix = self.load_weighted_graph(
    #             adjacency_filename
    #         )

    #         th = 0
    #         self.adjacency_matrix[self.adjacency_matrix <= th] = 0

    #     else:
    #         adjacency_filename += ".json"

    #         # ext = filename.split(".")[1]
    #         # assert ext in ["csv", "json"]
    #         # assert ext == "json"

    #         adj_mat = self.load_adjacency_matrix_json(adjacency_filename)

    #         # This makes sure that there are no self-edges (diagonal is 0s)
    #         if self_edges == False:
    #             np.fill_diagonal(adj_mat, 0)

    #         self.adjacency_matrix = adj_mat

    #     print("Adjacency matrix loaded!")

    def split_adjacency_matrix(self):
        """ """
        adj_mat_obj_occ = self.adjacency_matrix.copy()
        adj_mat_obj_occ[: self.n_out_classes :, :] = 0.0
        adj_mat_obj_occ[:, : self.n_out_classes :] = 0.0

        adj_mat_bipartite = self.adjacency_matrix.copy()
        adj_mat_bipartite[self.n_out_classes :, self.n_out_classes :] = 0.0

        # self.adj_mat_obj_occ = adj_mat_obj_occ.copy()
        # self.adj_mat_bipartite = adj_mat_bipartite.copy()
        return adj_mat_obj_occ, adj_mat_bipartite
