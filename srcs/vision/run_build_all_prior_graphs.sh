#!/bin/bash
#
##############################################################################
# Authors: 
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/02/02
# Modified Date: 2023/03/14
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
#-----------------------------------------------------------------------------
#
# run as:
# source run_prior_graph_builder.sh <model_name> <fold_id> <n_privacy_classes> <k>
##############################################################################
#
##############################################################################
MODEL_NAME=cgpa

# K=5
# source run_prior_graph_builder.sh $MODEL_NAME 0 2 $K
# source run_prior_graph_builder.sh $MODEL_NAME 1 2 $K
# source run_prior_graph_builder.sh $MODEL_NAME 2 2 $K

# source run_prior_graph_builder.sh $MODEL_NAME 0 3 $K
# source run_prior_graph_builder.sh $MODEL_NAME 1 3 $K
# source run_prior_graph_builder.sh $MODEL_NAME 2 3 $K


# K=3
# source run_prior_graph_builder.sh $MODEL_NAME 0 2 $K
# source run_prior_graph_builder.sh $MODEL_NAME 1 2 $K
# source run_prior_graph_builder.sh $MODEL_NAME 2 2 $K

# source run_prior_graph_builder.sh $MODEL_NAME 0 3 $K
# source run_prior_graph_builder.sh $MODEL_NAME 1 3 $K
# source run_prior_graph_builder.sh $MODEL_NAME 2 3 $K

K=3
source run_prior_graph_builder.sh $MODEL_NAME 99 2 $K

##############################################################################