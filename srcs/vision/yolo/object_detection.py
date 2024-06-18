#!/usr/bin/env python
#
# Python script for training Graph Privacy Advisor with a trainer class.
#
##############################################################################
# Authors:
# - Dimitrios Stoidis, dimitrios.stoidis@qmul.ac.uk
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#
#  Created Date: 2023/02/09
# Modified Date: 2023/09/21
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
import argparse

import time
import datetime

import random

import json
import csv

import numpy
import pandas as pd

from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from srcs.vision.yolo.models import *

from srcs.utils import (
    device,
    bbox_iou
    )

from pdb import set_trace as bp

Tensor = torch.cuda.FloatTensor
#############################################################################

def write_bounding_boxes():
    _set = ["train_", "test_"]
    database = "PicAlert"
    for s in _set:
        print("Detections for Set: ", s)
        public_file = open("./data_preprocess_/data_files/public_IPD.txt", "r")
        pub_file = public_file.readlines()
        private_file = open(
            "./data_preprocess_/data_files/private_IPD.txt", "r"
        )
        priv_file = private_file.readlines()

        temp_file = open(database + s + "debug.txt", "w")
        [temp_file.write(file1) for file1 in pub_file]
        [temp_file.write(file2) for file2 in priv_file]
        temp_file.close()
        temp_file = open(database + s + "debug.txt", "r")
        images = temp_file.readlines()
        prev_time = time.time()

        print("Number of images Images:{}".format(len(images)))
        detectedObjectsFile = open(s + database + "bboxes_debug.csv", "w")
        writer = csv.writer(detectedObjectsFile)

        for i, file in enumerate(images):
            filename = file.split("/")[-1].split(".")[0]
            try:
                img = Image.open(file.strip()).convert("RGB")
                detections = detect_image(img)
                print("image ", filename)

                img = np.array(img)
                pad_x = max(img.shape[0] - img.shape[1], 0) * (
                    img_size / max(img.shape)
                )
                pad_y = max(img.shape[1] - img.shape[0], 0) * (
                    img_size / max(img.shape)
                )
                unpad_h = img_size - pad_y
                unpad_w = img_size - pad_x
                categs, obj_categ, bboxes = [], [], []

                if detections is not None:
                    print("Detections...")
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)

                    # browse detections and draw bounding boxes
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                        bb = [
                            x1.item(),
                            y1.item(),
                            box_w.item(),
                            box_h.item(),
                        ]  # bb = [x1.item(), y1.item(), x2.item(), y2.item()]
                        bboxes.append(bb)
                        categs.append(int(cls_pred))
                        obj_categ.append(classes[int(cls_pred)])
                        print(classes[int(cls_pred)])
                        breakpoint()
                    row = [filename, obj_categ]
                    writer.writerow(row)

                    generate_json(categs, bboxes, filename, s, database)
                else:
                    print("Background \n")
                    row = [filename, "background"]
                    writer.writerow(row)
                    bb = [[0, 0, img.shape[0], img.shape[1]]]
                    generate_json([80], bb, filename, s, database)
            except:
                pass

            print("[%] {:.3f}".format((i * 100) / len(images)))
        detectedObjectsFile.close()

        inference_time = datetime.timedelta(seconds=time.time() - prev_time)
        print("Inference Time: %s" % inference_time)


def write_detections_confidence():
    with open(
        "./data_preprocess_/data_files/public_IPD.txt", "r"
    ) as public_file, open(
        "./data_preprocess_/data_files/private_IPD.txt", "r"
    ) as private_file:
        pub_file = public_file.readlines()
        priv_file = private_file.readlines()
        images = [file1.strip() for file1 in pub_file] + [
            file2.strip() for file2 in priv_file
        ]
    with open(
        "./data_preprocess_/data_files/ipd_data_manifest_multiclass.csv", "r"
    ) as file:
        reader = csv.reader(file)
        images_train_val = [row[0] for row in reader if row[1] != "2"]
    print("Number of images Images:{}".format(len(images_train_val)))
    detectedObjectsFile = open(
        "./resources/annotations/detections_confidence.csv", "w"
    )
    writer = csv.writer(detectedObjectsFile)

    for i, file in enumerate(images):
        filename = file.split("/")[-1].split(".")[0]
        img = Image.open(file).convert("RGB")
        if filename in images_train_val:
            detections = detect_image(img)
            print("image ", filename)
            categs, obj_categ, confidences = [], [], []
            if detections is not None:
                print("Detections...")
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)

                # browse detections and draw bounding boxes
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    categs.append(int(cls_pred))
                    confidences.append(conf)
                    obj_categ.append(classes[int(cls_pred)])
                    print(
                        classes[int(cls_pred)],
                        cls_pred.detach().cpu(),
                        n_cls_preds,
                        conf.detach().cpu(),
                    )

                row = [
                    filename,
                    ", ".join(obj_categ),
                    [round(tensor.item(), 4) for tensor in confidences],
                ]
                writer.writerow(row)
            else:
                print("No detections -- background")
                row = [filename, "no detection", 0.0]
                writer.writerow(row)
            print("[%] {:.3f}".format((i * 100) / len(images)))

    detectedObjectsFile.close()


#############################################################################


def non_max_suppression(
    prediction, num_classes, conf_thres=0.5, nms_thres=0.4
):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    Code originally from https://pjreddie.com/darknet/yolo/.
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)

    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2

    # Why is this inverted
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf.float(), class_pred.float()), 1
        )
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(
                detections_class[:, 4], descending=True
            )
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data

            # Add max detections to outputs
            output[image_i] = (
                max_detections
                if output[image_i] is None
                else torch.cat((output[image_i], max_detections))
            )

    return output


def load_coco_classes(root_dir, config):
    """
    Loads class labels at 'path'
    Code originally from https://pjreddie.com/darknet/yolo/.
    """
    fp = open(os.path.join(root_dir, config["yolo_paths"]["path"], 
        config["yolo_paths"]["class_file"]), "r")
    names = fp.read().split("\n")[:-1]
    fp.close()

    return names


#############################################################################


class YoloDetector(object):
    """ """

    def __init__(self, args, config):
        self.__version__ = "v3"

        self.config = config

        self.repo_dir = args.root_dir
        self.imgname_datadir_prefix = args.data_dir

        # hyper-parameters in yolov3
        self.conf_th = config["yolo_params"]["confidence_thresh"]  
        self.nms_th = config["yolo_params"]["nms_thresh"] 

        # Size of the image
        self.img_sz = config["yolo_params"]["image_size"] 

        config_path = os.path.join(
            self.repo_dir, config["yolo_paths"]["path"], 
            config["yolo_paths"]["config_file"]
            )
        weights_path = os.path.join(
            self.repo_dir, config["yolo_paths"]["path"], 
            config["yolo_paths"]["weights_file"]
            )
        self.check_weights_file(weights_path)
        self.load_model(config_path, weights_path)

        self.classes = load_coco_classes(self.repo_dir, config)

        # Number of object categories (for COCO=80, no background)
        self.n_obj_cls = config["yolo_params"]["num_obj_cat"]

        # Maximum number of Region of Interest (ROIs) to be detected
        self.max_n_rois = config["yolo_params"]["max_num_rois"]


        # self.dataset = args.dataset

        # # self.img_l = self.get_image_paths_manifest()
        # if self.dataset == "PrivacyAlert":
        #     self.img_l = self.get_image_list_privacy_alert()
        # elif self.dataset == "GIPS":
        #     self.img_l = self.get_image_list_privacy_gips()

    def check_weights_file(self, weights_path):
        if not os.access(weights_path, os.W_OK):
            weight_url = os.path.join(
                "https://pjreddie.com/media/files/yolov3.weights",
            )

            print(
                "wget -P "
                + os.path.join(self.repo_dir, self.config["yolo_paths"]["path"])
                + " --no-check-certificate "
                + weight_url
            )
            os.system(
                "wget -P "
                + os.path.join(self.repo_dir, self.config["yolo_paths"]["path"])
                + " --no-check-certificate "
                + weight_url
            )

    def get_image_list_privacy_gips(self):
        """ """
        print("Loading GIPS annotation file ...")

        df = pd.read_csv(
            os.path.join(
                self.imgname_datadir_prefix,
                "imglist.txt",
            ),
            delimiter=",",
            header=None,
            index_col=False,
        )

        l_imgs = df[0].to_list()

        return l_imgs

    def get_image_list_privacy_alert(self):
        """ """
        print("Loading Privacy Alert annotation file ...")

        # df = pd.read_csv(
        #     os.path.join(
        #         self.data_dir,
        #         "annotations",
        #         "labels.csv",
        #     ),
        #     delimiter=",",
        #     index_col=False,
        # )

        df = pd.read_csv(
            os.path.join(
                self.imgname_datadir_prefix,
                "imglist.txt",
            ),
            delimiter=",",
            header=None,
            index_col=False,
        )

        l_imgs = df[0].to_list()

        return l_imgs

    def get_image_paths_manifest(self):
        """
        Function partially taken from dataset.py
        """
        print("Loading image filenames from manifest ...")
        df = pd.read_csv(
            os.path.join(
                self.repo_dir,
                "resources",
                "privacy",
                "annotations",
                "ipd_data_manifest.csv",
            ),
            delimiter=",",
            index_col=False,
        )

        l_imgs = df["Image Name"].values

        # As filenames alone cannot help retrieve the images, we append the
        # full filepath and extension. We therefore create an updated list
        json_out = os.path.join(
            self.repo_dir,
            "resources",
            "privacy",
            "annotations",
            "ipd_imgs.json",
        )
        annotations = json.load(open(json_out, "r"))["annotations"]

        print(
            "Number of images (VISPR + PicAlert): {:d}".format(
                len(annotations)
            )
        )
        print("Number of annotated images: {:d}".format(l_imgs.shape[0]))

        l_img_ann = [x["image"] for x in annotations]

        new_img_l = []
        for idx, img in enumerate(tqdm(l_imgs, ascii=True)):
            idx2 = l_img_ann.index(img)

            full_img_path = annotations[idx2]["fullpath"]
            new_img_l.append(full_img_path)

        print("... loaded images!")

        return new_img_l

    def load_model(self, config_path, weights_path):
        """
        Load DarkNet model and weights
        Darknet model originally from https://pjreddie.com/darknet/yolo/
        """
        self.model = Darknet(config_path, img_size=self.img_sz)
        self.model.load_weights(weights_path)
        
        self.model.to(device)
        self.model.eval()

    def detect_image(self, img):
        """
        Detect objects for a single image after transforming and converting the
        image into a PyTorch tensor.

        As the model is run on a single image, the
        output of the non-maximum suppression is a list with a single element,
        and hence we return only the element 0 of the list as output of the
        function.

        Input:
            - img: as a Numpy array.

        Output:
            - detections: PyTorch tensor of dimensionality Cx7, where C is the
                          number of localised objects ih the current image.
        """
        # scale and pad image
        ratio = min(self.img_sz / img.size[0], self.img_sz / img.size[1])

        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)

        img_transforms = transforms.Compose(
            [
                transforms.Resize((imh, imw)),
                transforms.Pad(
                    (
                        max(int((imh - imw) / 2), 0),
                        max(int((imw - imh) / 2), 0),
                        max(int((imh - imw) / 2), 0),
                        max(int((imw - imh) / 2), 0),
                    ),
                    (128, 128, 128),
                ),
                transforms.ToTensor(),
            ]
        )

        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)

        input_img = Variable(image_tensor.type(Tensor))

        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(
                detections, self.n_obj_cls, self.conf_th, self.nms_th
            )

        return detections[0]

    def convert_detections_format(self, detections, w, h):
        """ """
        classes = load_coco_classes(self.repo_dir, self.config)
        assert len(classes) == self.n_obj_cls

        categs = []
        obj_categ = []
        bboxes = []
        confidences = []

        # bboxes_14 = torch.zeros((self.max_n_rois, 4))

        if detections is not None:
            # print("Detections...")

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            pad_x = max(h - w, 0) * (self.img_sz / max((w, h)))
            pad_y = max(w - h, 0) * (self.img_sz / max((w, h)))
            unpad_h = self.img_sz - pad_y
            unpad_w = self.img_sz - pad_x

            # browse detections and draw bounding boxes
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_h = ((y2 - y1) / unpad_h) * h
                box_w = ((x2 - x1) / unpad_w) * w

                y1 = ((y1 - pad_y // 2) / unpad_h) * h
                x1 = ((x1 - pad_x // 2) / unpad_w) * w

                bb = [
                    x1.item(),
                    y1.item(),
                    box_w.item(),
                    box_h.item(),
                ]  # bb = [x1.item(), y1.item(), x2.item(), y2.item()]

                bboxes.append(bb)
                categs.append(int(cls_pred))
                obj_categ.append(classes[int(cls_pred)])
                confidences.append(round(conf.item(), 4))

        bboxes_objects = {
            "bboxes": bboxes, 
            "categories" : categs,
            "confidence" : confidences,
            }

        bboxes_14 = self.convert_bboxes(bboxes_objects, 
            self.max_n_rois, w, h, self.img_sz)

        bboxes_cat_conf_tensor = self.prepare_cat_conf(bboxes_objects, self.max_n_rois)

        return bboxes_objects, bboxes_cat_conf_tensor, bboxes_14

    def convert_bboxes(self, bboxes_objects, max_n_rois, img_w, img_h, rescaled_size):
        """Convert the bounding boxes with respect to the scaled image.


        Arguments:
            - bboxes_objects:
            - max_n_rois:
            - img_w:
            - img_h:
            - rescaled_size:

        Return:
            - bboxes_14: rescaled bounding boxes
        """

        bboxes = torch.Tensor(bboxes_objects["bboxes"])

        bboxes_14 = torch.zeros((max_n_rois, 4))

        # Retain only the first N ROIs if the number of detected instances is
        # larger than the maximum
        if bboxes.size()[0] > max_n_rois:
            print(
                "Warning! Number of detected ROIs is larger than the set maximum!"
            )
            bboxes = bboxes[:max_n_rois]

        if bboxes.size()[0] != 0:
            # re-scale, image size is wxh so change bounding boxes dimensions from wxh space to 448x448 range
            bboxes[:, 0::4] = rescaled_size / img_w * bboxes[:, 0::4]
            bboxes[:, 1::4] = rescaled_size / img_h * bboxes[:, 1::4]
            bboxes[:, 2::4] = rescaled_size / img_w * bboxes[:, 2::4]
            bboxes[:, 3::4] = rescaled_size / img_h * bboxes[:, 3::4]

            bboxes_14[0 : bboxes.size(0), :] = bboxes

        return bboxes_14


    def prepare_cat_conf(self, bboxes_objects, max_n_rois):
        """Prepare the Tensor with the confidence and category for each instance.

        Arguments:
            - bboxes_objects:
            - max_n_rois: maximum number of regions of interests (RoI)

        Return:
            - bboxes_cat_conf:  a (N+1)x2 Tensor where the first column is the
                                category of each object instance and the
                                second columns is its confidence. N is the
                                number of object instances. The tensor contains
                                also a first row (header) with the number N.
        """
        n_objs = len(bboxes_objects["categories"])

        bboxes_cat_conf = torch.zeros(max_n_rois + 1, 2)

        bboxes_cat_conf[:, 0] = -1
        bboxes_cat_conf[0, :] = n_objs

        bboxes_cat_conf[1 : n_objs + 1, 0] = torch.IntTensor(
            bboxes_objects["categories"]
        )

        bboxes_cat_conf[1 : n_objs + 1, 1] = torch.Tensor(
            bboxes_objects["confidence"]
        )

        return bboxes_cat_conf

    def save_image_bbox(self, detections, img, image_fn):
        """ Save image with coloured bounding boxes overlaid.
        """
        classes = load_coco_classes(self.repo_dir, self.config)
        n_obj_cats = len(classes)

        # Convert RGB to BGR
        img = np.array(img)
        open_cv_image = img[:, :, ::-1].copy()

        pad_x = max(img.shape[0] - img.shape[1], 0) * (
            self.img_size / max(img.shape)
        )
        pad_y = max(img.shape[1] - img.shape[0], 0) * (
            self.img_size / max(img.shape)
        )
        unpad_h = self.img_size - pad_y
        unpad_w = self.img_size - pad_x

        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            # n_cls_preds = len(unique_labels)

            cmap = plt.get_cmap("nipy_spectral")
            # colors = [cmap(i) for i in np.linspace(0, 1, n_obj_cats)]

            colors = []
            for i in np.linspace(0, 1, n_obj_cats):
                x = cmap(i)
                y = tuple([int(z * 255) for z in x])
                colors.append(y)
                bbox_colors = colors
            # bbox_colors = random.sample(colors, n_obj_cats)

            fig, ax = plt.subplots(1)
            # browse detections and draw bounding boxes
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                y1_n = int(y1 - pad_y // 2 / unpad_h * img.shape[0])
                x1_n = int(x1 - pad_x // 2 / unpad_w * img.shape[1])

                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])

                y2_n = y1_n + box_h
                x2_n = x1_n + box_w

                color = bbox_colors[int(cls_pred)]

                cv2.rectangle(
                    open_cv_image, (x1_n, y1_n), (x2_n, y2_n), color, 2
                )
                text = "{}: {:.2f}".format(classes[int(cls_pred)], conf)
                cv2.putText(
                    open_cv_image,
                    text,
                    (int(x1_n), int(y1_n) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
        
        fullpath = os.path.join(self.data_dir, "imgs_det")
        if not os.path.exists(os.path.dirname(fullpath)):
            os.makedirs(os.path.dirname(fullpath), exist_ok=True)

        cv2.imwrite(
            os.path.join(fullpath, image_fn.split("/")[-1]), open_cv_image
        )

    def save_bboxes_to_json(self, bboxes_categories, filename, data_dir):
        """
        """
        json_file = json.dumps(bboxes_categories)

        path_elems = filename.split("/")
        if "batch" in path_elems[-2]:
            fullpath = os.path.join(
                data_dir,
                "dets",
                path_elems[-2],
                path_elems[-1].split(".")[0] + ".json",
            )
        else:
            fullpath = os.path.join(
                data_dir,
                "dets",
                path_elems[-1].split(".")[0] + ".json",
            )

        if not os.path.exists(os.path.dirname(fullpath)):
            os.makedirs(os.path.dirname(fullpath), exist_ok=True)

        fh = open(fullpath, "w")
        fh.write(json_file)
        fh.close()

    def load_image(self, filename):
        if filename.endswith("\n"):
            filename = filename.split("\n")[0]

        fullpath = os.path.join(self.imgname_datadir_prefix, "imgs", filename)

        try:
            img = Image.open(fullpath).convert("RGB")
            # print("Processing image " + filename)
        except:
            print("Image " + filename + " not existing!")

        return img

    def get_bboxes_images(self):
        """ """
        # out_filename = os.join.path(self.repo_dir, "resources", "annotations", "detections_confidence_diff_08_07.csv")
        # fh = open(out_filename, "r")
        # reader = csv.reader(fh)
        # image_name = [row[0] for row in reader]
        # fh.close()

        for idx, file in enumerate(tqdm(self.img_l, ascii=True)):
            filename = self.img_l[idx]
            img = self.load_image(filename)

            detections = self.detect_image(img)

            categs = []
            # obj_categ = []
            bboxes = []
            confidences = []

            if detections is not None:
                img = np.array(img)
                pad_x = max(img.shape[0] - img.shape[1], 0) * (
                    self.image_size / max(img.shape)
                )
                pad_y = max(img.shape[1] - img.shape[0], 0) * (
                    self.image_size / max(img.shape)
                )
                unpad_h = self.image_size - pad_y
                unpad_w = self.image_size - pad_x

                # browse detections and draw bounding boxes
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                    bb = [x1, y1, box_w, box_h]

                    bboxes.append(bb)
                    confidences.append(round(conf.item(), 4))
                    categs.append(int(cls_pred))
                    # obj_categ.append(classes[int(cls_pred)])

            bboxes_categories = {"categories": [], "bboxes": [], "confidence": []}

            bboxes_categories["categories"] = categs
            bboxes_categories["bboxes"] = bboxes
            bboxes_categories["confidence"] = confidences

            self.save_bboxes_to_json(bboxes_categories, filename)

        print("Saved bounding boxes for each image!")

    def run(self):
        self.get_bboxes_images()
        # get_one_image_bb(det, image)


#############################################################################


def GetParser():
    parser = argparse.ArgumentParser(
        description="YOLOv3 Detector (Pre-trained on COCO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--data_dir", type=str, default=".")

    parser.add_argument("--dataset", type=str, default="PrivacyAlert")

    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join("resources", "privacy", "obj_det", "yolov3.cfg"),
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=os.path.join(
            "resources", "privacy", "obj_det", "yolov3.weights"
        ),
    )
    parser.add_argument(
        "--class_path",
        type=str,
        default=os.path.join("resources", "privacy", "obj_det", "coco.names"),
    )

    parser.add_argument("--img_sz", type=int, default=416)
    parser.add_argument("--conf_th", type=float, default=0.8)
    parser.add_argument("--nms_th", type=float, default=0.4)

    return parser


#############################################################################

if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))
    print("PyTorch {}".format(torch.__version__))
    # print('OpenCV {}'.format(cv2.__version__))

    if device == "cuda":
        torch.cuda.set_device(0)

    print("Using {}".format(device))

    # Arguments
    parser = GetParser()
    args = parser.parse_args()

    yolo_det = YoloDetector(args)
    yolo_det.run()
