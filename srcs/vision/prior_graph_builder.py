#!/usr/bin/env python
#
# <Brief description here>
#
##################################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/03/09
# Modified Date: 2023/03/09
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
# Parent class for building Prior Knowledge Graphs


class PriorKnowlegeGraphBuilder(object):
    def __init__(self, args):
        self.root_dir = args.root_dir
        self.data_dir = args.data_dir

        self.model_name = args.model_name

        self.partition = args.training_mode
        self.fold_id = args.fold_id
        self.partitions_fn = args.partitions_fn
        self.n_privacy_cls = args.n_privacy_cls

        self.get_list_imgs_training_fold()
        self.n_imgs = len(self.l_imgs)

        self.b_self_edges = args.self_edges

        self.Gnx = nx.Graph()

        self.graph_d = None

        self.n_edges_added = 0

        self.n_nodes = 2

    def set_nodes_info(self):
        self.n_nodes_2 = self.n_nodes * self.n_nodes
        self.max_n_edges = self.n_nodes * (self.n_nodes - 1) / 2.0
        self.edges_factor = self.n_nodes_2 / 64.0

    def get_image_list_privacy_alert(self, img_name):
        """ """
        df1 = pd.read_csv(
            self.partitions_fn.replace("_splits", ""),
            delimiter=",",
            index_col=False,
        )
        batches = df1["batch"]
        batches = "batch" + batches.astype(str)

        image_list = batches.astype(str) + "/" + img_name.astype(str)
        # image_list = batches.astype(str) +"/"+ img_name.astype(str) + ".jpg"

        return image_list

    def get_list_imgs_training_fold(self):
        """ """
        df = pd.read_csv(
            self.partitions_fn,
            delimiter=",",
            index_col=False,
        )

        img_name = self.get_image_list_privacy_alert(df["Image Name"])
        # img_name = df["Image Name"]

        labels = df["Label {:d}-class".format(self.n_privacy_cls)]

        if self.partition == "final":
            fold_str = "Final"
        elif self.partition == "original":
            fold_str = "Original"
        else:
            fold_str = "Fold {:d}".format(self.fold_id)

        # The prior knowledge graph must be built only from the training data
        # (index 0 in this case)
        l_train_imgs = img_name[df[fold_str] == 0].values
        l_train_labels = labels[df[fold_str] == 0].values

        ### THIS IS FOR THE FULL IPD DATASET
        # # As filenames alone cannot help retrieve the images, we append the
        # # full filepath and extension. We therefore create an updated list
        # json_out = os.path.join(
        #     self.root_dir, "resources", "annotations", "ipd_imgs.json"
        # )
        # annotations = json.load(open(json_out, "r"))["annotations"]
        # l_img_ann = [x["image"] for x in annotations]

        # print("Loading training data ...")

        # idx_valid = np.where(l_train_labels != -1)[0].tolist()
        # l_train_labels = l_train_labels[idx_valid]

        # new_img_l = []
        # for idx, img in enumerate(tqdm(l_train_imgs)):
        #     if idx not in idx_valid:
        #         continue

        #     idx2 = l_img_ann.index(img)
        #     full_img_path = annotations[idx2]["fullpath"]
        #     new_img_l.append(full_img_path)

        #     # for elem in annotations:
        #     #     if elem["image"] == img:
        #     #         full_img_path = elem["fullpath"]
        #     #         new_img_l.append(full_img_path)

        # self.l_imgs = new_img_l
        self.l_imgs = l_train_imgs
        self.l_labels = l_train_labels

        return l_train_imgs.tolist(), l_train_labels

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

        if self.partition == "final":
            graph_fn = "prior_graph_{:s}-final{:s}.json".format(
                self.model_name, suffix
            )
        elif self.partition == "original":
            graph_fn = "prior_graph_{:s}-original{:s}.json".format(
                self.model_name, suffix
            )
        else:
            graph_fn = "prior_graph_{:s}-{:d}{:s}.json".format(
                self.model_name, self.fold_id, suffix
            )

        fullpath = os.path.join(
            self.data_dir,
            "graph_data",
            "adj_mats",
            graph_fn,
        )

        return fullpath

    def load_graph_nx(self):
        fullpath = self.get_graph_fn()
        ## Check if file exists
        self.compute_graph_nx(fullpath)

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
        print("Number of graph nodes: {:d}".format(self.Gnx.number_of_nodes()))
        print("Number of graph edges: {:d}".format(self.Gnx.number_of_edges()))

        print(
            "Total number of possible edges (symmetric, not self-loops): {:d}".format(
                int(self.max_n_edges)
            )
        )
        print("Total number of possible edges: {:d}".format(self.n_nodes_2))
        print("Real number of nodes: {:d}".format(self.n_nodes))

        print("Number of edges added: {:d}".format(self.n_edges_added))

        self.is_graph_sparse()


#############################################################################


class PriorKnowlegeGraphGIP(PriorKnowlegeGraphBuilder):
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
                "resources",
                "obj_det",
                # img_name.split(".")[0] + ".json",
                img_name + ".json",
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
            "resources",
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


#############################################################################


class PriorKnowlegeGraphGPA(PriorKnowlegeGraphBuilder):
    def __init__(self, args):
        super().__init__(args)

        self.n_nodes = args.n_obj_cats + self.n_privacy_cls
        self.set_nodes_info()

        print(self.model_name)

    def get_graph_edges_from_files(self):
        """
        Compute the edges of the graphs with only COCO object categories as nodes.
        Detected objects are retrieved from the input file.
        """
        prior_graph = dict()
        for n in range(self.n_nodes):
            prior_graph[n] = []

        missing_object_img = []

        for img_idx in tqdm(range(self.n_imgs)):
            img_name = self.l_imgs[img_idx]

            fullpath = os.path.join(
                self.data_dir,
                "dets",
                # img_name.split(".")[0] + ".json",
                str(img_name) + ".json",
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

            img_cats = np.array(objs["categories"]) + self.n_privacy_cls
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

        print(
            "Number of missing object images: {:d}".format(
                len(missing_object_img)
            )
        )

    def run_compute_graph(self):
        self.get_graph_edges_from_files()
        self.print_graph_stats()
        self.save_graph_as_json()

    def run_graph_analysis(self):
        self.load_graph_nx()
        self.print_graph_stats()
        # self.save_graph_node_degrees()


#############################################################################
class PriorKnowlegeGraphCGPA(PriorKnowlegeGraphBuilder):
    def __init__(self, args):
        super().__init__(args)

        self.n_obj_cats = args.n_obj_cats
        self.n_scene_cls = args.n_scene_cats

        self.n_nodes = args.n_obj_cats + args.n_scene_cats
        self.set_nodes_info()

        self.node_labels = pd.read_csv(
            os.path.join(self.root_dir, "resources", "node_categories.csv")
        )
        idx = self.node_labels["category type"] == 0
        assert self.n_scene_cls == self.node_labels[idx].shape[0]

        self.top_k = (
            args.k
        )  # Number of scene tags to retained when building the graph

        self.scene_freq = np.zeros([self.n_scene_cls])

    def get_graph_edges_from_files(self, scene_probs_fn):
        df = pd.read_csv(
            scene_probs_fn, delimiter=";", index_col=False, header=None
        )
        probs_ipd = df.iloc[:, 1:].values * 100
        df = correct_filenames_scene(df)

        n_imgs_scene, n_scene_cls = probs_ipd.shape

        print(
            "Number of images: {:d} | {:d}".format(n_imgs_scene, self.n_imgs)
        )

        prior_graph = dict()
        for n in range(self.n_nodes):
            prior_graph[n] = []

        missing_scene_img = []
        missing_object_img = []

        for img_idx in tqdm(range(self.n_imgs)):
            img_name = self.l_imgs[img_idx]

            fullpath = os.path.join(
                self.root_dir,
                "resources",
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

            ind_obj = np.array(objs["categories"]) + self.n_scene_cls
            ind_obj_unique = np.unique(ind_obj)

            try:
                idx_s = df[
                    df.iloc[:, 0] == os.path.basename(img_name.split(".")[0])
                ].index.values[0]
            except:
                print("Missing scene image: {:s}".format(img_name))
                missing_scene_img.append(img_name)
                continue

            prob_j = probs_ipd[idx_s, :]
            sorted_prob_j = prob_j
            ind_scene = np.argsort(sorted_prob_j)[-args.k :]

            self.scene_freq[ind_scene] += 1

            ind = np.concatenate([ind_scene, ind_obj_unique])
            all_edges_dir = list(itertools.combinations(np.sort(ind), 2))

            # Self-edges for object categories
            if self.b_self_edges:
                for idx in ind_obj_unique:
                    if ind_obj[ind_obj == idx].shape[0] > 1:
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

        print(
            "Number of missing scene images: {:d}".format(
                len(missing_scene_img)
            )
        )
        print(
            "Number of missing object images: {:d}".format(
                len(missing_object_img)
            )
        )

    def save_graph_node_degrees(self):
        """
        From the loaded graph, the function extracts the node degrees, sort them
        and save them to file with the associated categories names
        """
        degrees_v = dict(self.Gnx.degree())

        hist_degrees = np.zeros([self.n_nodes, 3])
        hist_degrees[:, 1] = np.array(range(self.n_nodes))
        hist_degrees = hist_degrees.astype(int)
        for k, v in degrees_v.items():
            hist_degrees[int(k), 2] = int(v)

        # hist_degrees = hist_degrees[1:,:]
        hist_deg_sorted = hist_degrees[hist_degrees[:, 2].argsort()]
        hist_deg_sorted[:, 0] = np.array(range(self.n_nodes))

        graph_fn = "prior_graph_fold{:d}_top{:d}_degrees.txt".format(
            self.fold_id, self.top_k
        )

        fullpath = os.path.join(
            "{:s}".format(self.root_dir),
            "resources",
            "adjacency_matrix",
            self.model_name,
            graph_fn,
        )

        fh = open(fullpath, "w")
        fh.write("x\tnode_id\tnode_name\tdegree\n")
        n_nodes_gnx = hist_deg_sorted.shape[0]

        l_cat_names = []

        for n in range(n_nodes_gnx):
            x = hist_deg_sorted[n, 0]
            node_id = hist_deg_sorted[n, 1]
            y = hist_deg_sorted[n, 2]
            l = self.node_labels["category name"][node_id]
            fh.write("{:d}\t{:d}\t{:s}\t{:d}\n".format(x, node_id, l, y))
            # fh.write('{:d}\t{:d}\t{:d}\n'.format(x,node_id,y))

            l_cat_names.append(l)

        fh.close()

        # print(l_cat_names)

        hh = np.histogram(
            hist_deg_sorted[:, 2],
            bins=np.linspace(0, N_NODES, N_NODES + 1) - 0.5,
            density=False,
        )

        print("Saved graph node degrees to file!")

    def save_graph_image(self):
        node_labels_d = self.node_labels.iloc[:, :2].to_dict()

        # G.add_nodes_from(node_labels_d)

        # nx.draw_shell(G, with_labels = True, node_size=4000)
        pn = nx.random_layout(self.Gnx)
        # print(pn)
        # bp()
        # print(
        #     {
        #         v: k
        #         for v, k in node_labels_d["category name"].items()
        #         if str(v) in pn
        #     }
        # )
        # nx.draw_random(G, pos=pn, with_labels=True, labels={v:k for v, k in node_labels_d['category name'].items() if str(v) in pn})
        nx.draw(
            self.Gnx,
            pos=pn,
            with_labels=True,
            labels={
                v: k
                for v, k in node_labels_d["category name"].items()
                if str(v) in pn
            },
        )

        # print(G.nodes(data=True))

        filename = "prior_graph_fold{:d}_top{:d}.png".format(
            self.fold_id, self.top_k
        )

        fullpath = os.path.join(
            "{:s}".format(self.root_dir),
            "resources",
            "adjacency_matrix",
            self.model_name,
            filename,
        )

        fig = plt.gcf()
        fig.set_size_inches((20, 11.25), forward=False)
        # plt.show()
        plt.savefig(fullpath, dpi=300)
        plt.close()

    def save_frequency_scene_tags(self):
        """
        Save to file the frequence of the scene category.
        """
        filename = "scene_freq_fold{:d}_top{:d}.txt".format(
            self.fold_id, self.top_k
        )

        fullpath = os.path.join(
            "{:s}".format(self.root_dir),
            "resources",
            "adjacency_matrix",
            self.model_name,
            filename,
        )

        freq_per = self.scene_freq / self.n_imgs * 100

        fn_out1 = open(fullpath, "w")
        fn_out2 = open(fullpath.replace(".txt", ".csv"), "w")

        fn_out1.write("id\tCategory\tCounting\tFrequency\n")
        fn_out2.write("id;Category;Counting;Frequency\n")

        for js in range(self.n_scene_cls):
            c_id = self.node_labels["id"][js]
            l = self.node_labels["category name"][js]

            fn_out1.write(
                "{:d}\t{:s}\t{:d}\t{:.2f}\n".format(
                    int(c_id), l, int(self.scene_freq[js]), freq_per[js]
                )
            )
            fn_out2.write(
                "{:d};{:s};{:d};{:.2f}\n".format(
                    int(c_id), l, int(self.scene_freq[js]), freq_per[js]
                )
            )

        fn_out1.close()
        fn_out2.close()

        print("Scene tags frequency saved to files!")

    def run_compute_graph(self, scene_probs_fn):
        self.get_graph_edges_from_files(args.scene_probs_fn)
        self.print_graph_stats()
        self.save_graph_as_json("_top{:d}".format(self.top_k))

    def run_graph_analysis(self):
        self.load_graph_nx()

        self.print_graph_stats()
        self.save_frequency_scene_tags()

        self.save_graph_node_degrees()
        # self.save_graph_image()


#############################################################################


def GetParser():
    parser = argparse.ArgumentParser(
        description="Prior Knowledge Graph Builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--data_dir", type=str, default=".")

    parser.add_argument(
        "--scene_probs_fn", type=str, default="scene_probs.csv"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="cgpa",
        choices=["gip", "gpa", "cgpa"],
    )

    parser.add_argument("--n_obj_cats", type=int, default=80)
    parser.add_argument("--n_scene_cats", type=int, default=365)
    parser.add_argument("--n_privacy_cls", type=int, default=2)

    parser.add_argument("-k", type=int, default=5)

    parser.add_argument("--fold_id", type=int, default=0)

    parser.add_argument(
        "--partitions_fn", type=str, default="ipd_data_manifest.csv"
    )

    parser.add_argument(
        "--self_edges",
        default=False,
        type=bool,
        help="include self-edges for object categories in the graph",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="compute",
        choices=["compute", "stats", "compute_stats"],
    )

    parser.add_argument(
        "--training_mode",
        type=str,
        default="final",
        choices=["final", "crossval", "original"],
        required=True,
        help="Choose to run K-fold cross-validation or train the final model (full training set without validation split)",
    )

    return parser


if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))

    # Arguments
    parser = GetParser()
    args = parser.parse_args()

    if args.model_name == "gip":
        g_builder = PriorKnowlegeGraphGIP(args)

        if args.mode == "compute":
            g_builder.run_compute_graph()
        elif args.mode == "stats":
            g_builder.run_graph_analysis()
        elif args.mode == "compute_stats":
            g_builder.run_compute_graph()
            g_builder.run_graph_analysis()

    elif args.model_name == "gpa":
        g_builder = PriorKnowlegeGraphGPA(args)

        if args.mode == "compute":
            g_builder.run_compute_graph()
        elif args.mode == "stats":
            g_builder.run_graph_analysis()
        elif args.mode == "compute_stats":
            g_builder.run_compute_graph()
            g_builder.run_graph_analysis()

    elif args.model_name == "cgpa":
        g_builder = PriorKnowlegeGraphCGPA(args)

        if args.mode == "compute":
            g_builder.run_compute_graph(args.scene_probs_fn)
        elif args.mode == "stats":
            g_builder.run_graph_analysis()
        elif args.mode == "compute_stats":
            g_builder.run_compute_graph(args.scene_probs_fn)
            g_builder.run_graph_analysis()
