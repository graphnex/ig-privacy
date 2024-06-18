#!/bin/bash
#
##############################################################################
# Authors: 
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2023/02/09
# Modified Date: 2023/06/28
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
# PARAMETERS
#
SEED=789 #789, 35, 369
#
# ROOT_DIR=$HOME/graphnex/repo/image-privacy-advisor
ROOT_DIR=/import/smartcameras-002/alessio/GraphNEx/GNN-architecture
#
# Directory where the dataset is stored
DATA_DIR=/import/smartcameras-002/Datasets/graphnex/image-privacy/curated-privacy-alert-dataset
#
IMAGE_SIZE=416
CONFIDENCE_TH=0.6
NMS_TH=0.4
#
##############################################################################
# LINES FOR SERVER - ALESSIO
# module load cuda/10.2
# module load anaconda3
#
##############################################################################
#
conda activate image-privacy
#
CUDA_VISIBLE_DEVICES=0 python yolo/object_detection.py                \
    --root_dir  $ROOT_DIR         \
    --data_dir  $DATA_DIR         \
    --img_sz    $IMAGE_SIZE       \
    --conf_th   $CONFIDENCE_TH    \
    --nms_th    $NMS_TH
#
#
conda deactivate
#
echo "Finished!"
#
