#!/usr/bin/env python
#
# Graph Reasoning Model
#
# Notes: Code inspired from https://github.com/guang-yanng/Image_Privacy
# and https://github.com/HCPLab-SYSU/SR
#
##################################################################################
# Authors:
# - Dimitrios Stoidis, dimitrios.stoidis@qmul.ac.uk
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2023/01/17
# Modified Date: 2023/06/27
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

import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from srcs.utils import check_if_square, check_if_symmetric
from srcs.utils import device

from pdb import set_trace as bp

#############################################################################


class GAT(nn.Module):
    def __init__(
        self,
        arg_batch_size=1,
        arg_n_nodes=445,
        arg_n_obj_cls=80,
        hidden_state_channel=4,
        output_channel=2,
    ):
        super(GAT, self).__init__()

        self.batch_sz = arg_batch_size

        self.n_nodes = arg_n_nodes
        self.n_obj_cls = arg_n_obj_cls
        self.n_scene_cls = arg_n_nodes - arg_n_obj_cls

        self.hidden_state_channel = hidden_state_channel
        self.output_channel = output_channel

        self.fc_x = nn.Linear(hidden_state_channel, output_channel)
        self.fc_y = nn.Linear(hidden_state_channel, output_channel)
        self.fc_2 = nn.Linear(output_channel, 1)

        self.initialize_weights()

        self.print_attributes()

    def set_adjacency_matrix(self, in_matrix):
        """ """
        # TODO: check the use of in and out matrix
        self.mask = Variable(
            torch.zeros(self.n_scene_cls, self.n_obj_cls),
            requires_grad=False,
        ).to(device)
        # tmp = in_matrix[
        # 0 : self.n_scene_cls, self.n_scene_cls :
        # ]  # reason in ggnn # same as adjacency matrix shape [2, 81]

        out_matrix = (
            in_matrix.t()
        )  # Transpose the matrix to fit the original convention
        self.mask[
            np.where(
                out_matrix[: self.n_scene_cls, self.n_scene_cls :].cpu() > 0
            )
        ] = 1  # 1 for valid connections between objects and classes


    def forward(self, flatten_aog_nodes, input_x):
        """

        Arguments:
            - flatten_aog_nodes: matrix of node features reshaped into a
                    1-dimensional vector. Shape: [batch size * num nodes, 1]
            - input_x: (???) Shape: [batch size * num nodes, 2]
        """
        # Reshape the flatten vector into batches of node features
        # Shape: [batch size,  num nodes, 1]
        fan = flatten_aog_nodes.view(self.batch_sz, self.n_nodes, -1)

        # Reshape the variable fan into a matrix whose second dimension
        # correspond to the dimensionality of the hidden state
        # h_node are the special or scene nodes
        s_node = (
            fan[:, 0 : self.n_scene_cls]
            .contiguous()
            .view(-1, self.hidden_state_channel)
        )

        # map the scene node features to the number of out classes
        h_fcx = torch.tanh(self.fc_x(s_node))  # shape = [40, 512]

        # extend the previous features for the number of object classes
        x_node_enlarge = (
            h_fcx.contiguous()
            .view(self.batch_sz * self.n_scene_cls, 1, -1)
            .repeat(1, self.n_obj_cls, 1)
        )

        # Reshape the variable fan into a matrix whose second dimension
        # correspond to the dimensionality of the hidden state
        # o_node are the special or scene nodes
        o_node = (
            fan[:, self.n_scene_cls :]
            .contiguous()
            .view(-1, self.hidden_state_channel)
        )

        # map the object node features to the number of out classes
        h_fcy = torch.tanh(self.fc_y(o_node))  # shape = [1620, 512]

        # extend the previous features for the number of scene classes
        y_node_enlarge = (
            h_fcy.contiguous()
            .view(self.batch_sz, 1, self.n_obj_cls, -1)
            .repeat(1, self.n_scene_cls, 1, 1)
        )  # shape = [bs, 2, 81, 512]

        # fuse class and objects hidden states with low-rank bilinear pooling
        h_xy_cat = (
            x_node_enlarge.contiguous().view(-1, self.output_channel)
        ) * (y_node_enlarge.contiguous().view(-1, self.output_channel))

        # attention coefficient eij = rfc2
        att_coeffs = self.fc_2(h_xy_cat)  # [3240, 1]

        # normalise coefficients with sigmoid -- aij = sigma(eij)
        att_coeffs_n = torch.sigmoid(att_coeffs)

        mask_enlarge = self.mask.repeat(self.batch_sz, 1, 1).view(-1, 1)
        masked_att_coeffs_n = att_coeffs_n * mask_enlarge  # [3240, 1]

        ########################################
        ggnn_feature = input_x.contiguous().view(
            self.batch_sz, self.n_nodes, -1
        )  # [bs, 83, 512]

        # privacy/scene outputs
        s_out = ggnn_feature[:, 0 : self.n_scene_cls, :]  # [bs, 2, 512]

        # object outputs
        o_out = ggnn_feature[:, self.n_scene_cls :, :]  # [bs, 81, 512]

        o_out_enlarge = (
            o_out.contiguous()
            .view(self.batch_sz, 1, -1)
            .repeat(1, self.n_scene_cls, 1)
            .view(-1, self.output_channel)
        )  # [3240, 512]

        # ggnn_feat_enlarge = (
        #     ggnn_feature.contiguous()
        #     .view(self.batch_sz, 1, -1)
        #     .repeat(1, self.n_nodes, 1)
        #     .view(-1, self.output_channel)
        # )  # [3240, 512]

        # aggregate norm. attention coeffs. aij and output features Ooi
        # weight features of object nodes (context nodes)
        weight_ggnn_feat = o_out_enlarge * masked_att_coeffs_n  # [3240, 512]
        weight_ggnn_feat = weight_ggnn_feat.view(
            self.batch_sz, self.n_scene_cls, self.n_obj_cls, -1
        )  # [bs, 2, 81, 512]

        # f = [Ori, ai1Oo1, ai2Oo2, ..., aiNOoN]
        output = torch.cat(
            (
                s_out.contiguous().view(
                    self.batch_sz, self.n_scene_cls, 1, -1
                ),
                weight_ggnn_feat,
            ),
            2,
        )  # [bs, 2, 82, 512] eq. 10

        return output

    def initialize_weights(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
        https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        Source:
        https://github.com/gordicaleksa/pytorch-GAT/blob/main/models/definitions/GAT.py
        """

        for m in self.fc_2.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.1)
                # m.bias.data.zero_()

                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    # nn.init.xavier_uniform_(m.bias.data)

        for m in self.fc_x.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()

                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    # nn.init.xavier_uniform_(m.bias.data)

        for m in self.fc_y.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()

                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    # nn.init.xavier_uniform_(m.bias.data)

    def set_batch_size(self, batch_size):
        self.batch_sz = batch_size

    def print_attributes(self):
        """Print all class attributes."""
        print()  # empty line
        print("GRM-GAT - Attributes")
        print(" - Hidden state size: {:d}".format(self.hidden_state_channel))
        print(" - Output size: {:d}".format(self.output_channel))

        print(" - Number of classes: {:d}".format(self.n_scene_cls))
        print(" - Number of objects: {:d}".format(self.n_obj_cls))
        print(" - Number of nodes: {:d}".format(self.n_nodes))

        print(" - Batch size: {:d}".format(self.batch_sz))


class GGNN(nn.Module):
    def __init__(
        self,
        hidden_state_channel=10,
        output_channel=5,
        time_step=3,
        n_out_cls=2,
        n_obj_cls=80,
        n_scene_cls=365,
        batch_size=1,
        b_bce=False,
    ):
        super(GGNN, self).__init__()

        self.time_step = time_step

        self.hidden_state_channel = hidden_state_channel
        self.output_channel = output_channel

        self.num_classes = n_out_cls

        self.n_obj_cls = n_obj_cls
        self.n_scene_cls = n_scene_cls

        # self.n_nodes = n_obj_cls + n_scene_cls
        self.n_nodes = n_obj_cls + self.num_classes

        self.batch_sz = batch_size

        self.b_bce = b_bce

        # Store multiplication for avoiding re-computation
        self.bs_nn = self.batch_sz * self.n_nodes

        # Fully connected layers
        hsc1 = hidden_state_channel
        hsc2 = hidden_state_channel * 2

        self.fc_eq3_w = nn.Linear(hsc2, hsc1)  # hsc2
        self.fc_eq3_u = nn.Linear(hsc1, hsc1)
        self.fc_eq4_w = nn.Linear(hsc2, hsc1)  # hsc2
        self.fc_eq4_u = nn.Linear(hsc1, hsc1)
        self.fc_eq5_w = nn.Linear(hsc2, hsc1)  # hsc2
        self.fc_eq5_u = nn.Linear(hsc1, hsc1)

        self.fc_output = nn.Linear(hsc2, output_channel)
        self.ReLU = nn.ReLU(True)

        self.print_attributes()

    def print_attributes(self):
        """Print all class attributes."""
        print("\nGRM-GGNN - Attributes")
        print(" - Hidden state size: {:d}".format(self.hidden_state_channel))
        print(" - Output size: {:d}".format(self.output_channel))
        print(" - Time step: {:d}".format(self.time_step))
        print(" - Number of classes: {:d}".format(self.num_classes))
        print(" - Number of objects: {:d}".format(self.n_obj_cls))
        print(" - Number of nodes: {:d}".format(self.n_nodes))
        print(" - Batch size: {:d}".format(self.batch_sz))

    def propagation_model(self, batch_aog_nodes):
        """
        batch_aog_node: [xx, num_nodes, hidden_state_channel]
        [8, 445, 5]
        """
        bs_nn = self.bs_nn
        num_imgs = self.batch_sz

        # if batch_aog_nodes.size()[0] < self.batch_sz:
        #     print(num_imgs)
        #     print(bs_nc)
        #     print(batch_aog_nodes.size()[0])
        #     bp()
        #     num_imgs = batch_aog_nodes.size()[0]
        #     bs_nc = num_imgs * node_num

        flatten_aog_nodes = batch_aog_nodes.view(bs_nn, -1)  # [4450, 4]]

        if self.batch_in_adj_mat.type() != batch_aog_nodes.type():
            self.batch_in_adj_mat = self.batch_in_adj_mat.type(
                batch_aog_nodes.type()
            )
            self.batch_out_adj_mat = self.batch_out_adj_mat.type(
                batch_aog_nodes.type()
            )

        # eq(2)
        # TO CHECK THE ORDER OF THIS
        try:
            av1 = torch.cat(
                (
                    torch.bmm(self.batch_in_adj_mat, batch_aog_nodes),
                    torch.bmm(self.batch_out_adj_mat, batch_aog_nodes),
                ),
                2,
            )  # shape = [bs, 83, 8196]
        except:
            bp()

        av = av1.view(bs_nn, -1)

        # eq(3) zv = sigma(Wav + Uhv)
        zv = torch.sigmoid(
            self.fc_eq3_w(av) + self.fc_eq3_u(flatten_aog_nodes)
        )

        # eq(4) rv = sigma(Wav + Uhv)
        rv = torch.sigmoid(
            self.fc_eq4_w(av) + self.fc_eq3_u(flatten_aog_nodes)
        )

        # eq(5)
        # hv = tanh(Wav + U(rv*hv))
        hv = torch.tanh(
            self.fc_eq5_w(av) + self.fc_eq5_u(rv * flatten_aog_nodes)
        )

        # hv = (1-zv) * hv + zv * hv
        flatten_aog_nodes = (
            1 - zv
        ) * flatten_aog_nodes + zv * hv  # shape = [bs*83, 4098]

        batch_aog_nodes = flatten_aog_nodes.view(
            num_imgs, self.n_nodes, -1
        )  # shape = [bs, 83, 4098]

        return batch_aog_nodes, flatten_aog_nodes

    def forward(self, x):
        # assert x.size()[0] == self.batch_sz # not possible

        # x [10, 1780]

        x_reshaped = x.view(-1, self.hidden_state_channel)  # [3560, 4]
        #
        batch_aog_nodes = x_reshaped.view(
            -1, self.n_nodes, self.hidden_state_channel
        )  # [10, 445, 4]

        # propagation process
        for t in range(self.time_step):
            batch_aog_nodes, flatten_aog_nodes = self.propagation_model(
                batch_aog_nodes
            )

        # final hidden state of all nodes {h1, h2, .., hv}
        output = torch.cat((flatten_aog_nodes, x_reshaped), 1)

        # node-level feature ov = o([hv, xv]) through fc layer
        output = self.fc_output(output)
        output = torch.tanh(
            output
        )  # shape = [bs*83, 2], previously -->  [bs*83, 512]

        return output, flatten_aog_nodes

    def split_adj_mat_into_income_outcome(self, adj_mat, mode=None):
        """Convert a square adjacency matrix into input and output matrices.

        Inputs:
            - adj_mat: the square adjacency matrix

        The function stores the income and outcome edge matrices as class members:
            self.in_matrix
            self.out_matrix
        """
        if mode == "GPA-DS":
            # see load_nodes() in https://github.com/smartcameras/GPA/blob/main/ggnn.py
            mat = adj_mat[self.num_classes:,self.num_classes:]

            d_row = mat.shape[0]
            d_col = self.num_classes
            
            in_matrix = np.zeros((d_row + d_col, d_row + d_col))
            in_matrix[:d_row, d_col:] = mat  # adjacency matrix is in the top right corner
            
            out_matrix = np.zeros((d_row + d_col, d_row + d_col))
            out_matrix[d_col:, :d_row] = mat.transpose()  # adjacency matrix transpose is in the bottom left corner

            self.in_matrix = in_matrix.astype(np.float32)
            self.out_matrix = out_matrix.astype(np.float32)

        else:

            assert check_if_square(adj_mat)

            self.in_matrix = np.tril(adj_mat).astype(np.float32)
            self.out_matrix = np.triu(adj_mat).astype(np.float32)

        # in_matrix = np.tril(adj_mat)
        # out_matrix = np.triu(adj_mat)

        # ggnn_adj_mat = np.hstack((in_matrix, out_matrix))

        # n_rows = adj_mat.shape[0]
        # n_cols = adj_mat.shape[1]

        # assert n_rows * 2 == n_cols

        # self.in_matrix = adj_mat[:, :n_rows].astype(np.float32)
        # self.out_matrix = adj_mat[:, n_rows:].astype(np.float32)

    def get_in_matrix(self):
        return self.in_matrix

    def set_batch_size(self, batch_size):
        self.batch_sz = batch_size
        self.bs_nn = self.batch_sz * self.n_nodes

        if self.b_bce:
            assert self.num_classes == 2
            self.bs_nc = self.batch_sz
        else:
            assert self.num_classes >= 2
            self.bs_nc = self.batch_sz * self.num_classes

        self.batch_in_adj_mat = (
            self.in_matrix.repeat(self.batch_sz, 1)
            .view(self.batch_sz, self.n_nodes, -1)
            .to(device)
        )
        self.batch_out_adj_mat = (
            self.out_matrix.repeat(self.batch_sz, 1)
            .view(self.batch_sz, self.n_nodes, -1)
            .to(device)
        )

    def set_adjacency_matrix(self, adj_mat, mode=None):
        """ """
        self.split_adj_mat_into_income_outcome(adj_mat, mode=mode)
        # self.node_num = self.in_matrix.size()[0]

        self.mask = Variable(
            torch.zeros(self.num_classes, self.n_obj_cls),
            requires_grad=False,
        ).to(device)
        tmp = self.in_matrix[
            0 : self.num_classes, self.num_classes :
        ]  # reason in ggnn # same as adjacency matrix shape [2, 81]
        self.mask[
            np.where(tmp > 0)
        ] = 1  # 1 for valid connections between objects and classes

        # does this mean that the adjacency matrix is not updated during training?
        self.in_matrix = Variable(
            torch.from_numpy(self.in_matrix), requires_grad=False
        ).to(device)
        self.out_matrix = Variable(
            torch.from_numpy(self.out_matrix), requires_grad=False
        ).to(device)

        self.batch_in_adj_mat = (
            self.in_matrix.repeat(self.batch_sz, 1)
            .view(self.batch_sz, self.n_nodes, -1)
            .to(device)
        )
        self.batch_out_adj_mat = (
            self.out_matrix.repeat(self.batch_sz, 1)
            .view(self.batch_sz, self.n_nodes, -1)
            .to(device)
        )


################################################################################
# Graph Reasoning Model (GGNN + GAT )
class GraphReasoningModel(nn.Module):
    def __init__(
        self,
        adjacency_matrix="",
        grm_hidden_channel=10,
        grm_output_channel=5,
        time_step=3,
        batch_size=1,
        n_obj_cls=80,
        n_scene_cls=365,
        n_out_class=2,
        attention=True,
        b_bce=False,
    ):
        super(GraphReasoningModel, self).__init__()

        self.batch_sz = batch_size
        self.num_class = n_out_class

        self.hidden_channel = grm_hidden_channel
        self.output_channel = grm_output_channel

        self.time_step = time_step

        self.n_obj_cls = n_obj_cls
        # self.n_scene_cls = n_scene_cls
        self.graph_num = n_obj_cls + n_out_class

        self.b_bce = b_bce
        self.bs_nc = self.batch_sz * self.num_class

        self.bs_nn = self.batch_sz * self.graph_num

        self.attn = attention

        self.ggnn = GGNN(
            hidden_state_channel=self.hidden_channel,
            output_channel=self.output_channel,
            time_step=self.time_step,
            n_out_cls=self.num_class,
            n_obj_cls=self.n_obj_cls,
            n_scene_cls=self.num_class,
            batch_size=self.batch_sz,
            b_bce=self.b_bce,
        )

        self.gat = GAT(
            arg_batch_size=self.batch_sz,
            arg_n_nodes=self.graph_num,
            arg_n_obj_cls=self.n_obj_cls,
            hidden_state_channel=self.hidden_channel,
            output_channel=self.output_channel,
        )

        # self.reshape_input = nn.Linear(
        #     (self.n_obj_cls + 1) * self.n_scene_cls, (self.graph_num)
        # )
        # The modified GAT seems to always return 2 at the end
        # self.reshape_input = nn.Linear(
        #     (self.graph_num + 1) * self.graph_num * 2, (self.graph_num)
        # )

        # The modified GAT seems to always return 2 at the end
        # self.reshape_input = nn.Linear(
        #     (self.n_obj_cls + 1) * self.output_channel, (self.graph_num)
        # )

    def forward(self, x):
        if x.shape[0] < self.batch_sz:
            self.set_batch_size(x.shape[0])

        ggnn_feature, fan = self.ggnn(x)

        if self.attn:
            gat_feats = self.gat(fan, ggnn_feature)

            # gat_feats_reshaped_1 = gat_feats.view(self.bs_nc, -1)
            # output = self.reshape_input(gat_feats_reshaped_1)

            output = gat_feats.view(self.bs_nc, -1)

        else:
            ggnn_feat_reshaped = ggnn_feature.view(
                self.batch_sz, self.graph_num, -1
            )
            ggnn_feat_cls = ggnn_feat_reshaped[:, : self.num_class, :]
            output = torch.reshape(ggnn_feat_cls, (self.bs_nc, 2))

        return output

    def set_batch_size(self, batch_size):
        self.batch_sz = batch_size

        if self.b_bce:
            assert self.num_class == 2
            self.bs_nc = self.batch_sz
        else:
            assert self.num_class >= 2
            self.bs_nc = self.batch_sz * self.num_class

        self.ggnn.set_batch_size(batch_size)
        self.gat.set_batch_size(batch_size)

    def set_adjacency_matrix(self, adj_mat, mode=None):
        """ """
        self.ggnn.set_adjacency_matrix(adj_mat, mode=mode)
        self.gat.set_adjacency_matrix(self.ggnn.get_in_matrix())
