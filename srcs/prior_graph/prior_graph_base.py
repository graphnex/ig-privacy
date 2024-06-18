#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/09/05
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

import os
import sys
import argparse  # Parser for command-line options, arguments and sub-commands

import pandas as pd  # Open-source data analysis and manipulation tool
import numpy as np  # Scientific computing

import itertools

from tqdm import tqdm  # smart progress meter for loops

import json  # Open standard file format to store data

import networkx as nx  # Software for complex networks (including graphs)
import matplotlib.pyplot as plt  # Visualisation tool

from srcs.datasets.wrapper import WrapperDatasets

from pdb import set_trace as bp  # For debugging and adding breakpoints

# ----------------------------------------------------------------------------
# Utilities


def correct_filenames_scene(df):
    img_fns = df.iloc[:, 0].tolist()

    l_imgs = []
    for fn in img_fns:
        new_fn = fn.split(".")[0]
        l_imgs.append(new_fn)

    df.iloc[:, 0] = l_imgs

    return df


#############################################################################
# Parent class for Prior Knowledge Graphs


class PriorKnowlegeGraphBase(object):
    def __init__(self, config):
        self.root_dir = config["paths"]["root_dir"]
        
        self.dataset = config["dataset"]

        try:
            with open(
                os.path.join(self.root_dir, "configs", "datasets.json")
            ) as f:
                datasets_config = json.load(f)

            self.data_dir = os.path.join(
                datasets_config["paths"]["data_prefix"],
                datasets_config["datasets"][self.dataset]["data_dir"],
            )
        except ValueError:
            print("Dataset configuration file not correctly loaded!")

        # All categories: -1. Private class: 0. Public class: 1
        # self.category = args.category_mode

        # self.model_name = args.model_name

        # self.partition = args.training_mode
        self.fold_id = config["params"]["fold_id"]
        # self.partitions_fn = args.partitions_fn

        self.n_out_classes = 2  # default value

        # self.get_list_imgs_training_fold()

        # self.b_self_edges = args.self_edges
        self.b_self_edges = True

        self.Gnx = nx.Graph()
        self.graph_d = None
        self.n_edges_added = 0
        self.n_graph_nodes = 2

    def set_nodes_info(self):
        self.n_graph_nodes_2 = self.n_graph_nodes * self.n_graph_nodes
        self.max_n_edges = self.n_graph_nodes * (self.n_graph_nodes - 1) / 2.0
        self.edges_factor = self.n_graph_nodes_2 / 64.0

    def get_dataset_img_labels(self, config, mode):
        """ """
        print("Loading the {:s} dataset ...".format(config["dataset"]))

        training_mode = config["params"]["training_mode"]

        data_wrapper = WrapperDatasets(
            root_dir=config["paths"]["root_dir"],
            data_dir=self.data_dir,
            num_classes=config["net_params"]["num_out_classes"],
            fold_id=config["params"]["fold_id"],
            n_graph_nodes=config["net_params"]["n_graph_nodes"],
            node_feat_size=config["net_params"]["node_feat_size"],
        )

        data_wrapper.load_split_set(
            config["dataset"],
            partition=training_mode,
            mode=mode,
            b_filter_imgs=config["b_filter_imgs"],
        )

        data_split = data_wrapper.get_data_split(mode)

        self.l_imgs = data_split.imgs
        self.l_labels = data_split.labels

        self.n_imgs = len(self.l_imgs)

        if config["dataset"] == "IPD":
            self.data_prefix = data_split.data_prefix

    def reset_n_self_edges(self):
        self.n_edges_added = 0

    def is_graph_sparse(self):
        if self.n_edges_added < self.edges_factor:
            print("Graph is sparse")
            return True

        elif (self.n_edges_added >= self.edges_factor) and (
            self.n_edges_added < self.max_n_edges
        ):
            print("Graph is almost sparse")
            return True

        else:
            print("Graph is dense")
            return False

    def get_graph_fn(self, suffix=""):
        """ """

        graph_fn = "prior_graph_f{:d}_c{:d}{:s}.json".format(
            self.fold_id, self.n_out_classes, suffix
        )

        fullpath = os.path.join(
            "{:s}".format(self.root_dir),
            "assets",
            "adjacency_matrix",
            self.model_name,
            graph_fn,
        )

        return fullpath

    def load_graph_nx(self, mode="file"):
        """ """
        assert mode in ["adj_mat", "file"]

        if mode == "adj_mat":
            self.Gnx = nx.from_numpy_array(self.adjacency_matrix)
        elif mode == "file":
            fullpath = self.get_graph_fn()
            ## Check if file exists
            self.compute_graph_nx(fullpath)

    def get_n_edges_th(self, a_th):
        """ """
        adj_mat = (self.adjacency_matrix > a_th).astype(int)

        adj_mat[:2, :] = (self.adjacency_matrix[:2, :] > 0).astype(int)
        adj_mat[:, :2] = (self.adjacency_matrix[:, :2] > 0).astype(int)

        G = nx.from_numpy_array(adj_mat)

        return G.number_of_edges(), nx.number_of_isolates(G)

    def compute_graph_nx(self, json_file_path):
        adj_l = json.load(open(json_file_path, "r"))

        edges_l = []

        for k, neighbours in adj_l.items():
            for v in neighbours:
                edges_l.append((k, str(v)))

        self.Gnx.add_edges_from(edges_l)

    def save_graph_as_json(self, suffix=""):
        """
        The function saves the input graph G into a file as JSON format.

        The input graph G is a Python dictionary with the list of nodes as keys
        and a list of node ids as value. This list represents the edges between
        the (key) node and the (value) nodes. The function assumes that the graph
        G is undirected, that is the corresponding adjacency matrix is symmetric
        and for each pair of nodes, the reverse order of the node ids is also
        an edge.

        JSON format is convenient when the graph is undirected, edges do not have
        weights (i.e., either 0 or 1), and their are sparse (i.e., the number of
        edges is << N(N-1)/2, where N is the number of nodes).
        """
        # if n_edges_added < 0.8 * N_EDGES:
        if self.is_graph_sparse():
            out_fn = self.get_graph_fn(suffix)
            print(out_fn)

            dirname = os.path.dirname(out_fn)
            print(dirname)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

            G = self.graph_d

            # Convert keys of G from int64 to int
            G = {int(k): [int(i) for i in v] for k, v in G.items()}

            with open(out_fn, "w") as json_file:
                json.dump(G, json_file, indent=4)

        print("Sparse graph saved to JSON file!")

    def print_graph_stats(self):
        # print("Number of graph nodes: {:d}".format(self.Gnx.number_of_nodes()))
        # print("Number of graph edges: {:d}".format(self.Gnx.number_of_edges()))

        print(
            "Total number of possible edges (symmetric, not self-loops): {:d}".format(
                int(self.max_n_edges)
            )
        )
        print(
            "Total number of possible edges: {:d}".format(self.n_graph_nodes_2)
        )
        print("Real number of nodes: {:d}".format(self.n_graph_nodes))

        print("Number of edges added: {:d}".format(self.n_edges_added))

        self.is_graph_sparse()

    ##########################################################################
    #### Adjacency matrix

    def print_adj_mat(self):
        print(self.adjacency_matrix)

    def get_adjacency_matrix(self):
        return self.adjacency_matrix

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

    def load_adjacency_matrix(
        self,
        data_dir,
        model_name,
        partition,
        fold_id,
        mode="square_sym",
        self_edges=False,
        a_th=0,
    ):
        """

        Arguments:
            - filename: name of the file where the adjacency matrix is stored and to load.
            - mode:
            - self_edges: boolean to include also self-edges in the graph.

        mode:
            - square_sym: a square and symmetric matrix representing an undirected graph
        """
        print()  # Empty line on the command line

        assert mode in [
            "square_sym",
            "ac",
        ]


        adj_mat_fullpath = os.path.join(
            self.data_dir,
            "graph_data",
            self.dirname,
            "adj_mat",
        )

        # if self.dirname == "obj_scene":
        #     adj_mat_fullpath = os.path.join(
        #         self.data_dir,
        #         "graph_data",
        #         self.dirname,
        #         "adj_mat",
        #     )
        # else:
        #     adj_mat_fullpath = os.path.join(
        #         self.data_dir,
        #         "graph_data",
        #         "{:d}-class".format(self.n_out_classes),
        #         model_name,
        #         "adj_mat",
        #     )

        if partition == "crossval":
            adj_mat_fn = "prior_graph_fold{:d}".format(fold_id)

        if partition == "final":
            adj_mat_fn = "prior_graph_final"

        if partition == "original":
            adj_mat_fn = "prior_graph_original"
            # adj_mat_fn = "prior_graph_fold{:d}".format(fold_id)

        adjacency_filename = os.path.join(adj_mat_fullpath, adj_mat_fn)

        if mode == "ac":
            adjacency_filename += "_ac.csv"
            # print(adjacency_filename)

            self.adjacency_matrix = self.load_weighted_graph(
                adjacency_filename
            )

            # self.adjacency_matrix[self.adjacency_matrix <= a_th] = 0

            adj_mat_obj_occ = self.adjacency_matrix.copy()
            adj_mat_obj_occ[
                : self.n_out_classes, :
            ] = 0.0  # set first rows to zero
            adj_mat_obj_occ[
                :, : self.n_out_classes
            ] = 0.0  # set first columns to zero

            adj_mat_obj_occ[adj_mat_obj_occ <= a_th] = 0

            tmp_idx = self.n_out_classes
            self.adjacency_matrix[tmp_idx:, tmp_idx:] = adj_mat_obj_occ[
                tmp_idx:, tmp_idx:
            ]

        else:
            adjacency_filename += ".json"

            # ext = filename.split(".")[1]
            # assert ext in ["csv", "json"]
            # assert ext == "json"

            adj_mat = self.load_adjacency_matrix_json(adjacency_filename)

            # This makes sure that there are no self-edges (diagonal is 0s)
            if self_edges == False:
                np.fill_diagonal(adj_mat, 0)

            self.adjacency_matrix = adj_mat

        print("Adjacency matrix loaded!")

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
