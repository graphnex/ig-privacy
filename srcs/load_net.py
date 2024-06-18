#!/usr/bin/env python
#
# Python class for loading models in an agnostic way
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/09/21
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

# Package modules
from srcs.baselines.pers_rule import PersonRule as PersRuleBaseline

from srcs.nets.MLP import MLP as MLPnet
from srcs.nets.ga_mlp import GraphAgnosticMLP as GraphAgnosticMLPnet
from srcs.nets.gpa import GraphPrivacyAdvisor as GPAnet


def PersonRule(net_params):
    return PersRuleBaseline(net_params)

def MLP(net_params):
    return MLPnet(net_params)

def GraphAgnosticMLP(net_params):
    return GraphAgnosticMLPnet(net_params)

def GPA(net_params):
    return GPAnet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        "PersonRule": PersonRule,
        "MLP": MLP,
        "GraphAgnosticMLP": GraphAgnosticMLP,
        "GPA": GPA,
    }

    return models[MODEL_NAME](net_params)
