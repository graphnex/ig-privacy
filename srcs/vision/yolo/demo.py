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
# Modified Date: 2023/02/09
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

import argparse

import time
from datetime import datetime
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Package modules
from srcs.utils import set_seed, load_coco_classes
from srcs.yolo.object_detection import YoloDetector

from pdb import set_trace as bp

#############################################################################
# CONSTANTS
#
device = "cuda" if torch.cuda.is_available() else "cpu"


#############################################################################
# Class for
#
class YOLOdetectionDemo:
    def __init__(self, args):
        self.image = args.image
        self.img_size = args.image_size
        self.out_dir = args.out_dir
        self.root_dir = args.root_dir

        self.b_imglist = args.b_imglist
        self.imglistfn = args.imglistfn

        self.yolo_det = YoloDetector(args)
        # --------------------------------------------

    def load_image(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )  # imagenet values

        full_im_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )  # what about horizontal flip

        img_pil = Image.open(self.image).convert("RGB")

        full_im = full_im_transform(
            img_pil
        )  # e.g; for index 10 full_im.shape = [3, 448, 448]

        return img_pil, full_im

    def detect_object_categories(self):
        categs = []
        obj_categ = []

        img_pil, full_im = self.load_image()

        # classes = load_coco_classes(self.root_dir)
        detections = self.yolo_det.detect_image(img_pil)

        # if detections is not None:
        #     # browse detections
        #     for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        #         categs.append(int(cls_pred))
        #         obj_categ.append(classes[int(cls_pred)])
        # else:
        #     categs.append(80)
        #     obj_categ.append("background")

        # print("\nDetected object categories:")
        # print(obj_categ)

        self.save_image_bbox(detections, img_pil)

    def save_image_bbox(self, detections, img):
        classes = load_coco_classes(self.root_dir)
        n_obj_cats = len(classes)

        img = np.array(img)
        # Convert RGB to BGR
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

            # fig, ax = plt.subplots(1)
            # browse detections and draw bounding boxes
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                # box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                # y1_n = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                # x1_n = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                # bp()
                # y1_n = int(((y1) / unpad_h * img.shape[0]) - pad_y)
                # x1_n = int(((x1) / unpad_w * img.shape[1]) - pad_x)
                # y2_n = int(((y2) / unpad_h * img.shape[0]) - pad_y)
                # x2_n = int(((x2) / unpad_w * img.shape[1]) - pad_x)

                y1_n = int(y1 - pad_y // 2 / unpad_h * img.shape[0])
                x1_n = int(x1 - pad_x // 2 / unpad_w * img.shape[1])

                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])

                # y2_n = int(y2 - pad_y // 2 / unpad_h * img.shape[0])
                # x2_n = int(x2 - pad_x // 2 / unpad_w * img.shape[1])
                y2_n = y1_n + box_h
                x2_n = x1_n + box_w

                # draw a bounding box rectangle and label on the frame
                # color = bbox_colors[
                #     int(np.where(unique_labels == int(cls_pred))[0])
                # ]
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
                # bbox = patches.Rectangle(
                #     (x1, y1),
                #     box_w,
                #     box_h,
                #     linewidth=2,
                #     edgecolor=color,
                #     facecolor="none",
                # )
                # ax.add_patch(bbox)
                # plt.text(
                #     x1,
                #     y1,
                #     s=classes[int(cls_pred)],
                #     color="white",
                #     verticalalignment="top",
                #     bbox={"color": color, "pad": 0},
                # )
        # plt.imshow(img)
        # plt.axis("off")

        # show the output image
        # cv2.imshow("Image", open_cv_image)
        # cv2.waitKey(0)
        cv2.imwrite(
            os.path.join(self.out_dir, self.image.split("/")[-1]),
            open_cv_image,
        )

        # save image
        # plt.savefig(
        #     os.path.join(self.out_dir, self.image.split('/')[-1]),
        #     bbox_inches="tight",
        #     pad_inches=0.0,
        # )
        # plt.show()

    def detect_multiple_images(self):
        l_imgs = []

        fh = open(self.imglistfn, "r")
        for x in tqdm(fh):
            # print(x)
            self.image = x.rstrip()
            self.detect_object_categories()

    def run(self):
        if self.b_imglist:
            self.detect_multiple_images()
        else:
            self.detect_object_categories()


#############################################################################
#


def GetParser():
    parser = argparse.ArgumentParser(
        description="YOLO Detector - Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--image", type=str, default="xyz.png")
    parser.add_argument("--imglistfn", type=str, default="xyz.txt")
    parser.add_argument("--out_dir", type=str, default=".")

    parser.add_argument(
        "--image_size", type=int, default=416
    )  # yolov3 hyper-parameter

    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--data_dir", type=str, default=".")

    parser.add_argument(
        "--config_path",
        type=str,
        default="resources/obj_det/yolov3.cfg",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="resources/obj_det/yolov3.weights",
    )
    parser.add_argument(
        "--class_path",
        type=str,
        default="resources/coco.names",
    )

    parser.add_argument("--img_sz", type=int, default=416)
    parser.add_argument("--conf_th", type=float, default=0.6)
    parser.add_argument("--nms_th", type=float, default=0.4)

    if sys.version_info[0] >= 3:
        if sys.version_info[1] < 9:
            parser.add_argument(
                "--b_imglist",
                action="store_true",
                help="Force to use cardinality information.",
            )
            parser.set_defaults(feature=False)
        else:
            parser.add_argument(
                "--b_imglist", action=argparse.BooleanOptionalAction
            )
    else:
        parser.add_argument(
            "--b_imglist",
            action="store_true",
            help="Force to use cardinality information.",
        )
        parser.add_argument(
            "--no-b_imglist", dest="b_imglist", action="store_false"
        )
        parser.set_defaults(feature=False)

    return parser


#############################################################################
#

if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))
    print("PyTorch {}".format(torch.__version__))
    print("Using {}".format(device))
    print("OpenCV {}".format(cv2.__version__))

    # Arguments
    parser = GetParser()
    args = parser.parse_args()

    set_seed(args.seed)

    yolo_det = YOLOdetectionDemo(args)
    yolo_det.run()
