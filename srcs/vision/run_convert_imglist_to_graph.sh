#!/bin/bash
#
# Editable bash script for the pre-processing of the images into node features.
#
# Run as:
# 	source run_converter_img_to_graph.sh <img_filename>
#
##############################################################################
# Authors: 
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/05/05
# Modified Date: 2023/05/05
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
##############################################################################
#
SEED=789
#
# Directory of the repository in the current machine/server
ROOT_DIR='/import/smartcameras-002/alessio/GraphNEx/GNN-architecture'
#
DATASET=GIPS
#
# Directory where the dataset is stored
# DATA_DIR='/import/smartcameras-002/Datasets/graphnex/image-privacy/curated-privacy-alert-dataset/imgs'
DATA_DIR=$ROOT_DIR/assets/privacy/gips
#
# Output directory where the model information will be saved
OUT_DIR=$DATA_DIR/node_feats
#
# Create directory structure in case it does not exists
mkdir -p $DATA_DIR/imgs
mkdir -p $OUT_DIR
#
# Graph parameters
NODE_FEATURE_SIZE=1
MAX_NUM_ROIS=50
#
# YOLO object detector parameters
OBJECT_CATEGORIES=80
IMAGE_SIZE=416
CONFIDENCE_TH=0.6
NMS_TH=0.4

# S2P Classifier parameters
N_OUT_CLASSES=2
SCENE_CATEGORIES=365
BACKBONE_ARCH='resnet50'
TOP_K=-1
SCENE_MODEL_PATH=$ROOT_DIR/models/vision/s2p
SCENE_MODEL_FN=best_acc_s2p-12-0-0.pth
#
#
# IMGLIST_FN=$1
IMGLIST_FN=$DATA_DIR/imglist.txt
#
##############################################################################
# LINES FOR QMUL EECS SERVER - ALESSIO
module load cuda/10.2
module load anaconda3
#
##############################################################################
#
source activate graphnex-gnn
#
CUDA_VISIBLE_DEVICES=0 python img_to_graph.py   \
    --seed                  $SEED               \
    --root_dir              $ROOT_DIR           \
    --data_dir              $DATA_DIR           \
    --out_dir               $OUT_DIR            \
    --dataset               $DATASET            \
    --node_feature_size     $NODE_FEATURE_SIZE  \
    --max_num_rois          $MAX_NUM_ROIS       \
    --n_out_classes         $N_OUT_CLASSES	    \
    --num_obj_cat           $OBJECT_CATEGORIES  \
    --image_size		    $IMAGE_SIZE         \
    --conf_th   			$CONFIDENCE_TH      \
    --nms_th    			$NMS_TH				\
    --n_scene_categories    $SCENE_CATEGORIES   \
    --backbone_arch         $BACKBONE_ARCH      \
    --top_k					$TOP_K				\
    --model_path            $SCENE_MODEL_PATH   \
    --model_filename        $SCENE_MODEL_FN     \
    --imglistfn				$IMGLIST_FN			\
    --b_imglist                                 \
    --use_bce                                   
    #--norm_feat  								\
    #--cardinality                               \
    #--person                                    \
    

#
#
conda deactivate
#
echo "Finished!"
#
