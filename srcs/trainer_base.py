#!/usr/bin/env python
#
# Parent/base class for training different machine learning models.
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2023/01/30
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

import os
import sys
import argparse

import time
from datetime import datetime

import numpy as np
np.set_printoptions(threshold=sys.maxsize, precision=4)

from tqdm import tqdm
import json

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

# Package modules
from srcs.perfmeas_tracker import PerformanceMeasuresTracker, AverageMeter
from srcs.datasets.wrapper import WrapperDatasets
from srcs.utils import (
    save_model,
    device,
    crossval_stats_summary,
    plot_learning_curves,
)
from srcs.logging_gnex import Logging

from pdb import set_trace as bp  # This is only for debugging


#
#############################################################################
# Base Class for the training of a model
#
class TrainerBaseClass(object):
    def __init__(self, config, args):
        # Paths
        self.root_dir = config["paths"]["root_dir"]
        
        use_case_dir = "privacy"

        self.out_dir = os.path.join(
            self.root_dir, 
            "trained_models",
            use_case_dir,
            args.dataset.lower()
            )
        self.data_dir = os.path.join(
            config["paths"]["data_prefix"],
            config["datasets"][args.dataset]["data_dir"]
            )

        # --------------------------------------------
        # Training parameters
        self.params = config["params"]

        # Boolean for using binary cross-entropy loss
        self.b_bce = args.use_bce 

        # Boolean for using the weighted loss
        self.b_use_weight_loss = args.weight_loss

        self.resume = self.params["resume"]
        self.resume_measure = self.params["measure"]

        self.num_workers = config["num_workers"]

        # --------------------------------------------
        # Model network
        self.n_out_classes = config["net_params"]["num_out_classes"]
        
        self.net = None        

    def initialise_checkpoint_dir(self, model_name, n_out_classes):
        """ """
        checkpoint_dir = os.path.join(
            self.out_dir,
            # "{:d}-class".format(n_out_classes),
            model_name,
        )
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.checkpoint_dir = checkpoint_dir

    def initialise_performance_trackers(self, dataset, n_out_classes):
        """Monitor the performance measures (classes)."""
        self.train_metrics_tracker = PerformanceMeasuresTracker(
            dataset=dataset, n_cls=n_out_classes
        )
        self.val_metrics_tracker = PerformanceMeasuresTracker(
            dataset=dataset, n_cls=n_out_classes
        )
        self.best_metrics_tracker = PerformanceMeasuresTracker(
            dataset=dataset, n_cls=n_out_classes
        )

    def configure_optimizer(self, net, config):
        """ """

        if config["params"]["optimizer"] == "SGD":
            self.optimizer = getattr(optim, config["params"]["optimizer"])(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=config["params"]["init_lr"],
                weight_decay=config["params"]["weight_decay"],
                momentum=config["params"]["momentum"],
            )

            self.scheduler = lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=config["params"]["init_lr"],
                max_lr=config["params"]["max_lr"],
                mode="triangular",
                cycle_momentum=False,
            )

            self.optimizer_name = "SGD"
            self.scheduler_name = "CyclicLR"

        elif config["params"]["optimizer"] == "Adam":
            # Configuration based on Benchmarking GNNs -
            # SuperPixel Graph Classification CIFAR10
            # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/configs/superpixels_graph_classification_GCN_CIFAR10_100k.json

            self.optimizer = getattr(optim, config["params"]["optimizer"])(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=config["params"]["init_lr"],
                weight_decay=config["params"]["weight_decay"],
            )

            if config["params"]["training_mode"] == "final":
                optim_mode = "max"
            else:
                optim_mode = "min"

            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=optim_mode,
                factor=config["params"]["lr_reduce_factor"],
                patience=config["params"]["lr_schedule_patience"],
                verbose=True,
            )

            self.optimizer_name = "Adam"
            self.scheduler_name = "ReduceLROnPlateau"
        else:
            self.optimizer = getattr(optim, config["params"]["optimizer"])(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=config["params"]["init_lr"],
                weight_decay=config["params"]["weight_decay"],
            )

            self.scheduler = lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=config["params"]["init_lr"],
                max_lr=config["params"]["max_lr"],
                mode="triangular",
                cycle_momentum=False,
            )

            self.optimizer_name = config["params"]["optimizer"]
            self.scheduler_name = "CyclicLR"

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

    def initialise_log(self, model_name, n_out_classes):
        """Create training log and initialise the preamble."""
        filename = self.get_filename(
            model_name, extension=".txt", suffix="_training_log"
        )

        self.log = Logging()
        self.log.initialise(os.path.join(self.checkpoint_dir, filename))
        self.log.write_preamble(model_name, n_out_classes)
        self.log.write_training_parameters(
            self.params["batch_size_train"],
            self.params["batch_size_val"],
            self.params["max_num_epochs"],
            self.params["fold_id"],
            self.params["training_mode"],
        )
        self.log.write_line()

    def compute_and_save_info(
        self, perf_tracker, cm_pred, cm_targets, mode="epoch"
    ):
        perf_tracker.compute_all_metrics(cm_targets, cm_pred)

        assert mode in ["batch", "epoch"]
        if mode == "batch":
            self.log.write_batch_info(perf_tracker)
        elif mode == "epoch":
            perf_tracker.print_metrics()
            perf_tracker.write_metrics_to_log(self.log.get_log())

    def compare_metrics_and_save_model(self, model_name, epoch, es):
        """ """
        ext = ".pth"
        if ((epoch % 1) == 0) or (epoch == self.params["max_num_epochs"] - 1):
            es += 1

            eval_measures = [
                "balanced_accuracy",
                "weighted_f1_score",
                "accuracy",
                "macro_f1_score",
            ]

            assert self.params["measure"] in eval_measures
            measure = self.params["measure"]

            val_measure = self.val_metrics_tracker.get_measure(measure)

            if self.best_metrics_tracker.is_classification_report():
                best_measure = self.best_metrics_tracker.get_measure(measure)
            else:
                best_measure = 0
                self.best_metrics_tracker.set_classification_report(
                    self.val_metrics_tracker.get_classification_report()
                )

            if val_measure >= best_measure:
                self.best_metrics_tracker.set_measure(measure, val_measure)

                tmp_str = "model at epoch: {:d}\n".format(epoch)

                if measure == "balanced_accuracy":
                    filename = "acc"
                    print("\nSaved best balanced accuracy (%) " + tmp_str)
                    self.log.write_info(
                        "\nSaved best balanced accuracy (%) " + tmp_str
                    )

                if measure == "weighted_f1_score":
                    filename = "weighted_f1"
                    self.log.write_info("\nBest W-F1 " + tmp_str)

                if measure == "accuracy":
                    filename = "acc"
                    print("\nSaved best UBA(%) " + tmp_str)
                    self.log.write_info("\nSaved best UBA(%) " + tmp_str)

                if measure == "macro_f1_score":
                    filename = "macro_f1"
                    self.log.write_info("\nBest UW-F1 " + tmp_str)

                save_model(
                    self.net,
                    val_measure,
                    self.checkpoint_dir,
                    self.get_filename(model_name, ext, filename + "_"),
                    mode="best",
                    epoch=epoch,
                )

                self.log.write_info("Saved model at epoch {:d}".format(epoch))

                es = 0

        return es

    def set_performance_measures_best_epoch(self, best_epoch):
        """ """
        best_metrics = {
            "Fold id": self.params["fold_id"],
            "Epoch": best_epoch,
            "P-T": self.train_metrics_tracker.get_precision_overall() * 100,
            "P-V": self.val_metrics_tracker.get_precision_overall() * 100,
            "BA-T": self.train_metrics_tracker.get_balanced_accuracy() * 100,
            "BA-V": self.val_metrics_tracker.get_balanced_accuracy() * 100,
            # "R-T": self.train_metrics_tracker.get_recall_overall() * 100,
            # "R-V": self.val_metrics_tracker.get_recall_overall() * 100,
            "UBA-T": self.train_metrics_tracker.get_accuracy() * 100,
            "UBA-V": self.val_metrics_tracker.get_accuracy() * 100,
            "wF1-T": self.train_metrics_tracker.get_weighted_f1score() * 100,
            "wF1-V": self.val_metrics_tracker.get_weighted_f1score() * 100,
            "MF1-T": self.train_metrics_tracker.get_macro_f1_score() * 100,
            "MF1-V": self.val_metrics_tracker.get_macro_f1_score() * 100,
        }

        return best_metrics

    def save_best_epoch_measures(self, best_metrics, model_name):
        """Save to file the performance measures at the best epoch at the end
        of the training.
        """

        fullpathname = os.path.join(
            self.checkpoint_dir, model_name + "_crossval_best.txt"
        )

        if os.path.isfile(fullpathname):
            fh = open(fullpathname, "a")
        else:
            fh = open(fullpathname, "w")
            fh.write(
                "Fold\tEpoch\tP-T\tP-V\tBA-T\tBA-V\tUBA-T\tUBA-V\twF-T\twF-V\tMF-T\tMF-V\n"
            )
            fh.flush()

        fh.write(
            "{:d}\t{:3d}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\n".format(
                best_metrics["Fold id"],
                best_metrics["Epoch"],
                best_metrics["P-T"],
                best_metrics["P-V"],
                best_metrics["BA-T"],
                best_metrics["BA-V"],
                best_metrics["UBA-T"],
                best_metrics["UBA-V"],
                best_metrics["wF1-T"],
                best_metrics["wF1-V"],
                best_metrics["MF1-T"],
                best_metrics["MF1-V"],
            )
        )
        fh.flush()

        fh.close()

        # Save stats of the cross-validation using the resume_measure
        out_stats_fn = os.path.join(
            self.checkpoint_dir, model_name + "_crossval_stats.csv"
        )
        crossval_stats_summary(
            fullpathname, out_stats_fn, self.resume_measure
        )

    def load_training_graph_data(self, 
            dataset_name, 
            net_params,
            b_filter_imgs=False
            ):
        """ """
        data_wrapper = WrapperDatasets(
            root_dir=self.root_dir,
            data_dir=self.data_dir,
            num_classes=self.n_out_classes,
            fold_id=self.params["fold_id"],
            graph_mode=net_params["graph_type"],
            n_graph_nodes=net_params["n_graph_nodes"],
            node_feat_size=net_params["node_feat_size"],
            # adj_mat_fn=self.adjacency_filename,
        )
        data_wrapper.load_split_set(
            dataset_name, 
            self.params["training_mode"], 
            "train",
            b_use_card=net_params["use_card"], 
            b_use_conf=net_params["use_conf"],
            b_filter_imgs=b_filter_imgs,
        )

        self.training_loader = DataLoader(
            data_wrapper.get_training_set(),
            batch_size=self.params["batch_size_train"],
            shuffle=True,
            num_workers=self.num_workers,
            # drop_last=True,
        )

        if (self.params["training_mode"] == "crossval") | (
            self.params["training_mode"] == "original"
        ):
            self.validation_loader = DataLoader(
                data_wrapper.get_validation_set(),
                batch_size=self.params["batch_size_val"],
                shuffle=True,
                num_workers=self.num_workers,
                # drop_last=False,
            )

        self.cls_weights = data_wrapper.get_class_weights()
        self.data_wrapper = data_wrapper

    def train_epoch(self, epoch):
        """Train 1 epoch of the model"""
        self.set_batch_size(self.params["batch_size_train"])

        self.net = self.net.to(device)
        self.net.train()

        # Initialise the instance of the average meter class to monitor the loss
        train_losses = AverageMeter()

        self.log.write_epoch_info("Training")

        # Initialise training
        start_epoch_time = time.time()
        start_batch_time = start_epoch_time

        cm_pred = []
        cm_targets = []

        for batch_idx, (
            node_feats,
            target,
            weights,
            adj_mat,
            image_name,
        ) in enumerate(tqdm(self.training_loader, ascii=True)):
            #  Take GT labels
            target_var = Variable(target).to(device, non_blocking=True)
            targets = target_var.data.cpu().numpy()

            batch_stats = np.array(
                [
                    len(targets[targets == x]) / len(targets) * 100
                    for x in range(self.n_out_classes)
                ]
            )

            if batch_stats.any() == 0.0:
                print("One class not sampled!")

            self.optimizer.zero_grad()

            outputs = self.net(node_feats.to(device), adj_mat)

            weights_var = Variable(weights)
            weights_var = weights_var.to(device).to(torch.float32)

            if self.b_bce:
                assert self.n_out_classes == 2

                target_var = target_var.to(torch.float32)

                assert len(outputs.shape) in [1, 2]

                if len(outputs.shape) == 1:
                    # This is for the GRM case and should go first to avoid errors in accessing the dimensionality
                    out_logits = outputs
                elif (len(outputs.shape) == 2) & (outputs.shape[1] == 1):
                    # This is for the GAT case
                    out_logits = outputs[:, 0]

                if self.b_use_weight_loss:
                    bce = torch.nn.BCEWithLogitsLoss(pos_weight=weights_var)
                else:
                    bce = torch.nn.BCEWithLogitsLoss()

                loss = bce(out_logits, target_var)

                out_probs = torch.sigmoid(out_logits)
                preds = out_probs.round().data.cpu().numpy().tolist()

            else:
                assert self.n_out_classes >= 2

                cls_weights_var = torch.from_numpy(self.cls_weights)
                cls_weights_var = cls_weights_var.to(device).to(torch.float32)

                if self.b_use_weight_loss:
                    ce_loss = torch.nn.CrossEntropyLoss(weight=cls_weights_var)
                else:
                    ce_loss = torch.nn.CrossEntropyLoss()

                loss = ce_loss(outputs, target_var)

                output_np = F.softmax(outputs, dim=1).data.cpu().numpy()

                preds = list(np.argmax(output_np, axis=1)) 

            loss.backward()
            self.optimizer.step()

            train_losses.update(loss.item())

            # Take predictions from Graph model
            cm_pred = np.concatenate([cm_pred, preds])
            cm_targets = np.concatenate([cm_targets, targets])

            if batch_idx % 20 == 0 and batch_idx > 1:
                self.compute_and_save_info(
                    self.train_metrics_tracker,
                    cm_pred,
                    cm_targets,
                    "batch",
                )

            start_batch_time = time.time()

        self.compute_and_save_info(
            self.train_metrics_tracker, cm_pred, cm_targets, "epoch"
        )

        # print(
        #     "Epoch processing time: {:.4f} seconds".format(
        #         time.time() - start_epoch_time
        #     )
        # )

        return train_losses.get_average()

    def val_epoch(self):
        """ """
        print("\nValidating ...")

        # self.set_batch_size(self.params["batch_size_val"])
        self.net.set_batch_size(self.params["batch_size_val"])

        self.net = self.net.to(device)
        self.net.eval()

        self.log.write_epoch_info("Validating")

        # Initialise the instance of the average meter class to monitor the loss
        val_losses = AverageMeter()

        # Initialise validation variables
        cm_pred = []
        cm_targets = []

        prediction_scores = []
        target_scores = []

        # Initialise training
        start_epoch_time = time.time()
        start_batch_time = start_epoch_time

        with torch.no_grad():
            for batch_idx, (
                node_feats,
                target,
                weights,
                adj_mat,
                image_name,
            ) in enumerate(tqdm(self.validation_loader, ascii=True)):
                target_var = Variable(target).to(device, non_blocking=True)

                outputs = self.net(node_feats.to(device), adj_mat)

                weights_var = Variable(weights)
                weights_var = weights_var.to(device).to(torch.float32)

                if self.b_bce:
                    assert self.n_out_classes == 2

                    assert len(outputs.shape) in [1, 2]

                    if len(outputs.shape) == 1:
                        out_logits = outputs
                    elif (len(outputs.shape) == 2) & (outputs.shape[1] == 1):
                        out_logits = outputs[:, 0]

                    # convert logits into probabilities
                    out_probs = torch.sigmoid(out_logits)

                    # Round the probabilities to determine the class
                    preds = out_probs.round()

                    # Convert predictions (tensor) into a list
                    preds = preds.data.cpu().numpy().tolist()
                    prediction_scores.append(preds)

                    if self.b_use_weight_loss:
                        bce = torch.nn.BCEWithLogitsLoss(
                            pos_weight=weights_var
                        )
                    else:
                        bce = torch.nn.BCEWithLogitsLoss()

                    target_var = target_var.to(torch.float32)

                    val_loss = bce(out_logits, target_var)

                else:
                    assert self.n_out_classes >= 2

                    cls_weights_var = torch.from_numpy(self.cls_weights)
                    cls_weights_var = cls_weights_var.to(device).to(
                        torch.float32
                    )

                    if self.b_use_weight_loss:
                        ce_loss = torch.nn.CrossEntropyLoss(
                            weight=cls_weights_var
                        )
                    else:
                        ce_loss = torch.nn.CrossEntropyLoss()

                    val_loss = ce_loss(outputs, target_var)

                    # Compute the predicted class for all data with softmax
                    outputs_np = F.softmax(outputs, dim=1).data.cpu().numpy()
                    preds = np.argmax(outputs_np, axis=1)
                    prediction_scores.append(outputs_np[:, 0])

                val_losses.update(val_loss.item())

                # Add the predictions in current batch to all predictions
                cm_pred = np.concatenate([cm_pred, preds])

                # Update targets (labels)
                targets = target_var.data.cpu().numpy()
                cm_targets = np.concatenate([cm_targets, targets])
                target_scores.append(targets)

                # img_arr.append(image_name)

                if batch_idx % 20 == 0 and batch_idx > 1:
                    self.compute_and_save_info(
                        self.val_metrics_tracker, cm_pred, cm_targets, "batch"
                    )

                # Reset the batch time
                start_batch_time = time.time()

            self.compute_and_save_info(
                self.val_metrics_tracker, cm_pred, cm_targets, "epoch"
            )

            # print(
            #     "Epoch processing time: {:.4f} seconds".format(
            #         time.time() - start_epoch_time
            #     )
            # )

        return val_losses.get_average()

    def train_and_val(self):
        t0 = time.time()

        start_epoch = 0
        es = 0
        file_mode = "w"

        ext = ".pth"

        model_name = self.net.get_model_name()

        num_epochs = self.params["max_num_epochs"]

        # acc_train = []
        # acc_val = []

        # Resume training for the new learning rate #
        if self.resume:
            start_epoch = self.load_checkpoint()
            file_mode = "a"

        filename = self.get_filename(
            model_name, suffix="_learning_curve", extension=".txt"
        )
        fh = open("{}/{}".format(self.checkpoint_dir, filename), file_mode)
        fh.write(
            "Epoch\tloss-T\tloss-V\tP-T\tP-V\tBA-T\tBA-V\tUBA-T\tUBA-V\twF-T\twF-V\tMF-T\tMF-V\n"
        )
        fh.flush()

        best_metrics = None
        for epoch in range(start_epoch, num_epochs):
            print("Epoch: {:d}/{:d}\n".format(epoch + 1, num_epochs))
            self.log.write_info(
                "Epoch: {:d}/{:d}\n".format(epoch, num_epochs)
            )
            sys.stdout.flush()

            loss_value = self.train_epoch(epoch)
            val_loss_value = self.val_epoch()

            es = self.compare_metrics_and_save_model(model_name, epoch, es)

            if self.scheduler_name == "CyclicLR":
                self.scheduler.step()

                my_lr = self.scheduler.get_last_lr()[
                    0
                ]  # This is only for CyclingLR
                print("Learning rate: {:.6f}".format(my_lr))
            elif self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(val_loss_value)

            if es == 0:
                best_metrics = self.set_performance_measures_best_epoch(epoch)

            fh.write(
                "{:3d}\t{:9.4f}\t{:9.4f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\n".format(
                    epoch,
                    loss_value,
                    val_loss_value,
                    self.train_metrics_tracker.get_precision_overall() * 100,
                    self.val_metrics_tracker.get_precision_overall() * 100,
                    self.train_metrics_tracker.get_balanced_accuracy() * 100,
                    self.val_metrics_tracker.get_balanced_accuracy() * 100,
                    self.train_metrics_tracker.get_accuracy() * 100,
                    self.val_metrics_tracker.get_accuracy() * 100,
                    self.train_metrics_tracker.get_weighted_f1score() * 100,
                    self.val_metrics_tracker.get_weighted_f1score() * 100,
                    self.train_metrics_tracker.get_macro_f1_score() * 100,
                    self.val_metrics_tracker.get_macro_f1_score() * 100,
                )
            )
            fh.flush()

            if self.optimizer.param_groups[0]["lr"] < self.params["min_lr"]:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

            # Stop training after params['max_time'] hours
            if time.time() - t0 > self.params["max_time"] * 3600:
                print("-" * 89)
                print(
                    "Max_time for training elapsed {:.2f} hours, so stopping".format(
                        self.params["max_time"]
                    )
                )
                break

        fh.close()

        if best_metrics:
            self.save_best_epoch_measures(best_metrics, model_name)

        save_model(
            self.net,
            self.params["measure"],
            self.checkpoint_dir,
            self.get_filename(model_name, ext, prefix="acc_"),
            mode="last",
            epoch=epoch,
        )
        self.log.write_info("Saved model at last epoch {:d}".format(epoch))

        outfilename = self.get_filename(
            model_name, suffix="_learning_curve", extension="_acc.png"
        )
        plot_learning_curves(
            self.checkpoint_dir,
            filename,
            outfilename,
            self.params["measure"],
            mode=self.params["training_mode"],
        )

        outfilename = self.get_filename(
            model_name, suffix="_learning_curve", extension="_loss.png"
        )
        plot_learning_curves(
            self.checkpoint_dir,
            filename,
            outfilename,
            "loss",
            mode=self.params["training_mode"],
        )

    def train_final_model(self):
        t0 = time.time()

        start_epoch = 0
        file_mode = "w"

        ext = ".pth"

        model_name = self.net.get_model_name()

        num_epochs = self.params["max_num_epochs"]

        # Resume training for the new learning rate #
        if self.resume:
            start_epoch = self.load_checkpoint()
            file_mode = "a"

        filename = self.get_filename(
            model_name, suffix="_learning_curve", extension=".txt"
        )
        fh = open("{}/{}".format(self.checkpoint_dir, filename), file_mode)
        fh.write("Epoch\tloss\tP-T\tBA-T\tUBA-T\twF-T\tMF-T\n")
        fh.flush()

        for epoch in range(start_epoch, num_epochs):
            print("Epoch: {:d}/{:d}\n".format(epoch, num_epochs))
            self.log.write_info(
                "Epoch: {:d}/{:d}\n".format(epoch, num_epochs)
            )
            sys.stdout.flush()

            loss_value = self.train_epoch(epoch)

            if self.scheduler_name == "CyclicLR":
                self.scheduler.step()

                my_lr = self.scheduler.get_last_lr()[0]
                print("Learning rate: {:.6f}".format(my_lr))

            elif self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(
                    self.train_metrics_tracker.get_balanced_accuracy() * 100
                )

            save_model(
                self.net,
                self.params["measure"],
                self.checkpoint_dir,
                self.get_filename(model_name, ext, prefix="acc_"),
                mode="last",
                epoch=epoch,
            )

            self.log.write_info("Saved model at epoch {:d}".format(epoch))

            fh.write(
                "{:3d}\t{:9.6f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\n".format(
                    epoch,
                    loss_value,
                    self.train_metrics_tracker.get_precision_overall() * 100,
                    self.train_metrics_tracker.get_balanced_accuracy() * 100,
                    self.train_metrics_tracker.get_accuracy() * 100,
                    self.train_metrics_tracker.get_weighted_f1score() * 100,
                    self.train_metrics_tracker.get_macro_f1_score() * 100,
                )
            )
            fh.flush()

            if self.optimizer.param_groups[0]["lr"] < self.params["min_lr"]:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

            if (self.scheduler_name == "CyclicLR") & (
                self.optimizer.param_groups[0]["lr"] > self.params["max_lr"]
            ):
                print("\n!! LR EQUAL TO MAX LR SET.")
                break

            # Stop training after params['max_time'] hours
            if time.time() - t0 > self.params["max_time"] * 3600:
                print("-" * 89)
                print(
                    "Max_time for training elapsed {:.2f} hours, so stopping".format(
                        params["max_time"]
                    )
                )
                break

        fh.close()

        outfilename = self.get_filename(
            model_name, suffix="_learning_curve", extension="_loss.png"
        )
        plot_learning_curves(
            self.checkpoint_dir, filename, outfilename, "loss", mode="final"
        )

    def load_checkpoint(self):
        print("==> Resuming from checkpoint..")

        ext = ".pth"
        fullpathname = os.path.join(
            self.checkpoint_dir,
            self.get_filename(
                self.net.get_model_name(), ext, prefix="last_acc_"
            ),
        )

        checkpoint = torch.load(fullpathname)
        self.net.load_state_dict(checkpoint["net"])
        self.val_metrics_tracker.set_measure(self.resume_measure, checkpoint["measure"])

        start_epoch = checkpoint["epoch"]

        return start_epoch

    def set_batch_size(self, batch_size):
        """ """
        self.batch_size = batch_size
        self.net.set_batch_size(batch_size)

    def run(self):
        """ """
        if (self.params["training_mode"] == "crossval") | (
            self.params["training_mode"] == "original"
        ):
            self.train_and_val()

        if self.params["training_mode"] == "final":
            self.train_final_model()

        self.log.write_ending()


#############################################################################
