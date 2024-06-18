#!/usr/bin/env python
#
# Python script for
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2023/08/09
# Modified Date: 2023/08/09
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

import os
import sys

import json

from PIL import Image

import torch
import torchvision.transforms as transforms

IMG_SZ = 448
#
#
full_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SZ, IMG_SZ)),  #  transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ]
)
#
# imagenet values
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

data_full_transform = transforms.Compose(
    [transforms.Resize((IMG_SZ, IMG_SZ)), transforms.ToTensor(), normalize]
)  # what about horizontal flip
#
##############################################################################
privacy_classes = {
    "PrivacyAlert": {
        "binary": {0: "public", 1: "private"},
    },
}


##############################################################################


def get_image_name(filename):
    """Get the image name without path and extension."""
    image_name = filename.split("/")[-1]  # e.g; '2017_80112549.json'
    image_name = image_name.split(".")[-2]

    return image_name


def load_image(filename, data_dir):
    """

    Return:
        - full_im: the loaded image in PIL format. If transformation is enabled,
            then the image is transformed and resized to a fixed dimensionality.
        - w: width of the original image (no transformations applied).
        - h: width of the original image (no transformations applied).
    """
    # For normalize
    # filename = self.imgs[index]
    if filename.endswith("\n"):
        filename = filename.split("\n")[0]

    filename = os.path.join(data_dir, filename)
    if not os.path.isfile(filename):
        print(filename)

    img = Image.open(filename).convert("RGB")
    (w, h) = img.size  # e.g; (1024, 1019)

    if data_full_transform is not None:
        full_im = data_full_transform(img)
        # e.g; for index 10 full_im.shape = [3, 448, 448]
    else:
        full_im = img

    return full_im, w, h


def get_bounding_box_filename(img_fn, bbox_dir):
    """ """
    path = img_fn.split("/")[-2:]  # e.g; ['train2017', '2017_80112549.jpg']

    if "VISPR" in img_fn:
        bbox_filename = os.path.join(
            bbox_dir,
            "VISPR",
            path[0],
            path[1].split(".")[0] + ".json",
        )
    else:
        bbox_filename = os.path.join(
            bbox_dir, path[0], path[1].split(".")[0] + ".json"
        )
    # path = os.path.join(self.anno_dir, path[0], path[1] + ".json")

    return bbox_filename


def load_bounding_boxes(img_fn, bbox_dir):
    """ """
    bbox_filename = get_bounding_box_filename(img_fn, bbox_dir)

    try:
        bboxes_objects = json.load(open(bbox_filename))
    except:
        print("File not found! {:s}".format(bbox_filename))
        # if self.partition == "train":
        #     bboxes_objects = json.load(open(path.replace("train", "test")))
        # else:
        #     try:
        #         bboxes_objects = json.load(
        #             open(path.replace("test", "train"))
        #         )
        #     except:
        #         bboxes_objects = json.load(
        #             open(path.replace("train", "test"))
        #         )
    return bboxes_objects
