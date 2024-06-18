#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/03/09
# Modified Date: 2023/08/15
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


class PriorKnowlegeGraphGIP(PriorKnowlegeGraphBase):
    def __init__(self, args):
        super().__init__(args)

        self.n_nodes = args.n_obj_cats + self.n_privacy_cls
        self.n_nodes_2 = self.n_nodes * self.n_nodes
        self.max_n_edges = self.n_nodes * (self.n_nodes - 1) / 2

        self.n_obj_cats = args.n_obj_cats

    def get_graph_edges_from_files(self):
        """
        Compute the edges of the graphs with only COCO object categories as nodes.
        Detected objects are retrieved from the input file.
        """
        # Private: 0; Public: 1 (according to the manifest)
        if self.n_privacy_cls == 2:
            print("Labels: Private: 0; Public: 1 (according to the manifest)")
            n_imgs_pri = len(self.l_labels[self.l_labels == 0].tolist())
            n_imgs_pub = len(self.l_labels[self.l_labels == 1].tolist())
        elif self.n_privacy_cls == 3:
            print(
                "Labels: Private: 0; Undecidable: 1; Public: 2 (according to the manifest)"
            )
            n_imgs_pri = len(self.l_labels[self.l_labels == 0].tolist())
            n_imgs_und = len(self.l_labels[self.l_labels == 1].tolist())
            n_imgs_pub = len(self.l_labels[self.l_labels == 2].tolist())
        else:
            print("Cannot handle number of classes different from 2 or 3!")
            return

        freq_mat = np.zeros([self.n_privacy_cls, self.n_obj_cats])

        missing_object_img = []

        for img_idx in tqdm(range(self.n_imgs)):
            img_name = self.l_imgs[img_idx]
            label = self.l_labels[img_idx]

            fullpath = os.path.join(
                self.root_dir,
                "assets",
                "obj_det",
                img_name.split(".")[0] + ".json",
            )

            try:
                objs = json.load(open(fullpath))
            except:
                # print("Missing object image: {:s}".format(fullpath))
                missing_object_img.append(img_name)
                continue

            if len(objs["categories"]) == 0:
                # print("Image with no detected objects: {:s}".format(fullpath))
                missing_object_img.append(img_name)
                continue

            img_cats = np.array(objs["categories"])

            # Even if the are multiple instances for the same category, the next
            # operation adds only 1
            # bp()
            print(len(img_cats))

            freq_mat[label, img_cats] += 1

        if self.n_privacy_cls == 2:
            freq_mat[0, :] /= n_imgs_pri
            freq_mat[1, :] /= n_imgs_pub
        elif self.n_privacy_cls == 3:
            freq_mat[0, :] /= n_imgs_pri
            freq_mat[1, :] /= n_imgs_und
            freq_mat[2, :] /= n_imgs_pub

        self.n_edges_added = np.count_nonzero(freq_mat)

        self.graph_d = freq_mat

        print(
            "Number of missing object images: {:d}".format(
                len(missing_object_img)
            )
        )

    def save_graph_as_csv(self):
        """
        Save the weighted, undirected, bipartite graph as an adjacency matrix.

        Given the special type of graph and simplicity, we can simply save a
        block of the adjacency matrix (relation between the public/private
        node with the object categories). For the .csv file, a row is an object
        category (80 COCO categories in total) and the two columns are the
        private and public labels of images. Each cell of the matrix provide
        the frequency ([0,1]) of the category with respect to the public/private
        label.
        """
        graph_fn = "prior_graph_f{:d}_c{:d}.csv".format(
            self.fold_id, self.n_privacy_cls
        )

        fullpath = os.path.join(
            "{:s}".format(self.root_dir),
            "assets",
            "adjacency_matrix",
            self.model_name,
            graph_fn,
        )

        dirname = os.path.dirname(fullpath)
        print(dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        if self.n_privacy_cls == 2:
            headers = ["Private", "Public"]
        elif self.n_privacy_cls == 3:
            headers = ["Private", "Undecidable", "Public"]

        pd.DataFrame(self.graph_d.transpose()).to_csv(
            fullpath, header=headers, index=None
        )

    def run_compute_graph(self):
        self.get_graph_edges_from_files()
        self.print_graph_stats()
        self.save_graph_as_csv()
