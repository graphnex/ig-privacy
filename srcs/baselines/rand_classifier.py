#!/usr/bin/env python
#
# <Brief description here>
#
################################################################################
# Authors:
# - Alessio Xompero
#
# Email: a.xompero@qmul.ac.uk
#
#  Created Date: 2023/01/24
# Modified Date: 2023/01/24
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

import sys
import argparse
import numpy as np
import datetime


class RandomClassifier:
    def __init__(self, arg_seed, arg_num_images, arg_num_classes):
        self.rnd_seed = arg_seed
        self.num_images = arg_num_images
        self.num_privacy_classes = arg_num_classes

        self.preds = np.zeros((self.num_images, 3))

    def predict_privacy_class(self):
        np.random.seed(self.rnd_seed)

        for x in range(0, self.num_images):
            rand_prob = np.random.rand()

            self.preds[x, 0] = x
            self.preds[x, 1] = rand_prob

            if self.num_privacy_classes == 2:
                pred = 1 if rand_prob > 0.5 else 0

            elif self.num_privacy_classes == 3:
                if rand_prob < 0.33:
                    pred = 0
                elif rand_prob >= 0.33 and rand_prob < 0.66:
                    pred = 1
                else:
                    pred = 2

            elif self.num_privacy_classes == 5:
                if rand_prob < 0.2:
                    pred = 0
                elif rand_prob >= 0.2 and rand_prob < 0.4:
                    pred = 1
                elif rand_prob >= 0.4 and rand_prob < 0.6:
                    pred = 2
                elif rand_prob >= 0.6 and rand_prob < 0.8:
                    pred = 3
                else:
                    pred = 4

            self.preds[x, 2] = pred

    def save_predictions(self, filename):
        myfile = open(filename, "w")
        myfile.write("ImageID,Privacy Probability,Privacy class\n")

        for x in range(0, self.num_images):
            myfile.write(
                "{:d},{:.2f},{:d}\n".format(
                    int(self.preds[x, 0]),
                    self.preds[x, 1],
                    int(self.preds[x, 2]),
                )
            )

        myfile.close()

    def save_classifier_info(self, filename):
        now = datetime.datetime.now()

        myfile = open(filename, "w")
        myfile.write("Random binary classifier for Image Privacy classification\n\n")
        myfile.write(now.strftime("%Y-%m-%d %H:%M:%S\n"))
        myfile.write("Seed: {:d}\n".format(self.rnd_seed))
        myfile.write("Number of images: {:d}\n".format(self.num_images))
        myfile.close()

    def run(self, filename):
        self.predict_privacy_class()
        self.save_classifier_info("log_rand_classifier.txt")
        self.save_predictions(filename)


def get_parser():
    parser = argparse.ArgumentParser(
        prog="Random classifier for Image Privacy classification",
        description="Random classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--filename", default="rand.csv", type=str)
    parser.add_argument("--num_images", default=684, type=int)
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--num_classes", type=int, default=2, choices=[2, 3, 5])

    return parser


if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))

    # Arguments
    parser = get_parser()
    args = parser.parse_args()

    rand_classifier = RandomClassifier(args.seed, args.num_images, args.num_classes)
    rand_classifier.run(args.filename)

    print("Finished!")
