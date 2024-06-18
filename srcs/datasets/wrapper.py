#!/usr/bin/env python
#
# Brief description here
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2023/05/05
# Modified Date: 2023/09/06
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

from srcs.datasets.privacyalert_graph import PrivacyAlertDataset

from pdb import set_trace as bp

#############################################################################
## Wrapper for multiple datasets


class WrapperDatasets(object):
    """Wrapper class to load a dataset"""

    def __init__(
        self,
        root_dir=".",
        data_dir="",
        graph_mode="obj_scene",
        # adj_mat_fn="",
        num_classes=2,
        fold_id=0,
        n_graph_nodes=82,
        node_feat_size=1,
    ):
        """Initialisation of the Wrapper Class

        If you change the input parameters, you would need to change it also in the
        function load_training_data() of the trainer_base file and in the
        load_testing_data() of the tester base
        """
        super(WrapperDatasets, self).__init__()

        self.root_dir = root_dir
        self.data_dir = data_dir

        self.n_out_classes = num_classes
        
        self.fold_id = fold_id
        
        # self.adjacency_filename = adj_mat_fn
        
        self.graph_mode = graph_mode
        self.n_graph_nodes = n_graph_nodes
        self.node_feature_size = node_feat_size

    def print_load_set(self, split_mode):
        """ """
        if split_mode == "train":
            print("\nLoading training set ...")

        if split_mode == "val":
            print("\nLoading validation set ...")

        if split_mode == "test":
            print("\nLoading testing set ...")

    def load_split_privacyalert(self, partition, split_mode, 
        _b_use_card, _b_use_conf, _b_filter_imgs):
        """ """

        self.print_load_set(split_mode)

        data_split = PrivacyAlertDataset(
            repo_dir=self.root_dir,
            data_dir=self.data_dir,
            partition=partition,
            split=split_mode,
            graph_mode=self.graph_mode,
            num_classes=self.n_out_classes,
            fold_id=self.fold_id,
            # adj_mat_fn=self.adjacency_filename,
            n_graph_nodes=self.n_graph_nodes,
            node_feat_size=self.node_feature_size,
            b_use_card=_b_use_card,
            b_use_conf=_b_use_conf,
            b_filter_imgs=_b_filter_imgs,
        )

        return data_split 

    def load_split_set(
        self,
        dataset_name,
        partition="final",
        mode="train",
        b_use_card=True,
        b_use_conf=False,
        b_filter_imgs=False,
    ):
        """

        - dataset_name
        - mode: string, either train or test
        """
        assert partition in ["crossval", "final", "original"]
        assert mode in ["train", "test"]

        # Privacy Alert dataset
        if dataset_name == "PrivacyAlert":
            if mode == "train":
                training_set = self.load_split_privacyalert(
                    partition, "train", b_use_card, b_use_conf, b_filter_imgs
                )

                if (partition == "crossval") | (partition == "original"):
                    validation_set = self.load_split_privacyalert(
                        partition, "val", b_use_card, b_use_conf, b_filter_imgs
                    )

            elif mode == "test":
                testing_set = self.load_split_privacyalert(
                    partition, "test", b_use_card, b_use_conf, b_filter_imgs
                )

        # Compute class weights for weighted loss in training
        if mode == "train":
            self.training_set = training_set

            if (partition == "crossval") | (partition == "original"):
                self.validation_set = validation_set

        elif mode == "test":
            self.testing_set = testing_set

        print()

    def get_training_set(self):
        return self.training_set

    def get_validation_set(self):
        return self.validation_set

    def get_testing_set(self):
        return self.testing_set

    def get_data_split(self, mode="test"):
        """ """
        assert mode in ["train", "val", "test"]

        if mode == "train":
            return self.training_set

        if mode == "val":
            return self.validation_set

        if mode == "test":
            return self.testing_set

    def get_class_weights(self):
        """Compute class weights for weighted loss in training."""
        return self.training_set.get_class_weights()

    def get_dataset_name(self, mode):
        assert mode in ["train", "val", "test"]

        if mode == "train":
            return self.training_set.get_name_low()

        if mode == "val":
            return self.validation_set.get_name_low()

        if mode == "test":
            return self.testing_set.get_name_low()

    def get_data_dir(self):
        return self.data_dir
