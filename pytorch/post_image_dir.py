"""
Takes directory of images and posts them to tensorboard
"""

import argparse
import numpy as np
import os
import shutil
import sys
from PIL import Image

import torch
import torchvision as tv

from tensorboardX import SummaryWriter

from pytorch.pyt_utils.utils import get_summary_writer

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def main(cfg):
    input_dir = os.path.abspath( cfg["input_dir"] )
    tboard_supdir = os.path.abspath( cfg["tboard_supdir"] )
    tboard_dir = pj(tboard_supdir, os.path.basename(input_dir))
    if pe(tboard_dir):
        if not cfg["force_overwrite"]:
            raise RuntimeError("Output directory %s already exists, use -f to "\
                    "overwrite" % tboard_dir)
        shutil.rmtree(tboard_dir)
    os.makedirs(tboard_dir)
    print("Created tensorboard logdir %s" % tboard_dir)

    writer = get_summary_writer(tboard_dir)
    T = tv.transforms.ToTensor()

    images = [f for f in os.listdir(input_dir) if f.endswith(cfg["image_ext"])]
    for i,img_name in enumerate(images):
        img = Image.open( pj(input_dir, img_name) )
        writer.add_image(img_name, T(img), i)
    print("%d images added." % len(images))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, required=True)
    parser.add_argument("-e", "--image-ext", type=str, default="")
    parser.add_argument("-o", "--tboard-supdir", type=str, 
            default=pj(HOME, "tensorboard"))
    parser.add_argument("-d", "--subdir-depth", type=int, default=1,
            help="Number of subdirectories deep to copy from input directory " \
                    "path to output directory path")
    parser.add_argument("-f", "--force-overwrite", action="store_true",
            help="If set, preexisting output directory will be deleted")
    cfg = vars( parser.parse_args() )
    main(cfg)

