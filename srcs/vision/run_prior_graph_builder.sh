#!/bin/bash
#
##############################################################################
# Authors: 
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/02/02
# Modified Date: 2023/06/08
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
# PARAMETERS
#
MODEL_NAME='gpa'
#
# Directory of the repository in the current machine/server
ROOT_DIR='/import/smartcameras-002/alessio/GraphNEx/GNN-architecture'
#
# $ROOT_DIR/resources/privacy/gips
DATA_DIR=/import/smartcameras-002/Datasets/graphnex/image-privacy/curated-privacy-alert-dataset
#
SCENE_PROBS_FN=$ROOT_DIR/results/scene_probs.csv
#
FOLD_ID=$1
#
# PARTITIONS_FN=$ROOT_DIR/resources/annotations/ipd_data_manifest.csv
PARTITIONS_FN=$DATA_DIR/annotations/labels_splits.csv
#
MODE='compute' # compute, stats, compute_stats
#
#
N_OBJECT_CATEGORIES=80
N_SCENE_CATEGORIES=365
N_PRIVACY_CLASSES=2
# 
K=5 #Number of top confident scene categories to retain
#
#
TRAINING_MODE=$2
##############################################################################
#
conda activate graphnex-gnn
#
python prior_graph_builder.py \
    --root_dir              $ROOT_DIR                 \
    --data_dir              $DATA_DIR                 \
    --model_name            $MODEL_NAME               \
    --scene_probs_fn        $SCENE_PROBS_FN           \
    -k                      $K                        \
    --n_obj_cats            $N_OBJECT_CATEGORIES      \
    --n_scene_cats          $N_SCENE_CATEGORIES       \
    --n_privacy_cls         $N_PRIVACY_CLASSES        \
    --fold_id               $FOLD_ID                  \
    --partitions_fn         $PARTITIONS_FN            \
    --mode                  $MODE                     \
    --training_mode         $TRAINING_MODE
    
    
#
#
conda deactivate
#
echo "Finished!"
#