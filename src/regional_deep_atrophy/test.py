#!/usr/bin/env python

"""
Script to train a Regional Deep Atrophy model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

Registration will be intra-subject scan-to-scan.
"""


# from comet_ml import Experiment
import os
import sys
import random
import argparse
import time
import math
import shutil
import numpy as np
import importlib
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from functools import partial
import pickle
from . import utils
from . import generators
from .utils import get_logger, get_tensorboard_formatter, RunningAverage
from . import layers
from . import networks
import matplotlib
import csv
from datetime import datetime
import torch.multiprocessing as mp

matplotlib.use('Agg')  # MUST BE CALLED BEFORE IMPORTING plt, or qt5agg for interactive plotting, Agg for non-interactive, such as saving to file


print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


def main(args):

    print("ROOT is ", args.ROOT)

    if not os.path.exists(args.ROOT):
        os.makedirs(args.ROOT)

    # extract shape from sampled input
    # inshape = (48, 80, 64)
    inshape = (args.input_D, args.input_H, args.input_W)
    print("inshape is", inshape)

    train_augment = ['normalize', 'rotate', 'random_crop'] # 'erase' , 'rotate', 'flip'
    # in training, flip is not needed because they are automatically included in rotation.
    eval_augment = ['normalize', 'fixed_crop']
    test_augment = ['normalize', 'fixed_crop'] # 'rotate', 
    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert np.mod(args.batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

    print("load model")
    model = networks.SingleAttentionALOHA.load(args.load_model, device)

    model.to(device)
    model.eval()
    ############################################################################

    # test on the test set
    if args.test_pairs is not None:

        print("=> Test on single image pair for Test Set")

        # prepare model directory
        model_dir = args.ROOT + "/test_test"
        try:
            shutil.rmtree(model_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
            print("Therefore, no need to remove the test directory.")

        os.makedirs(model_dir, exist_ok=True)

        # create logger
        logger = get_logger('RDA test for model' + model_dir, log_file=model_dir + '/test.log')
        logger.info(f'Start test on {device}.')

        test_dataset = generators.DatasetRISI3DPair(
                                    args.test_pairs,
                                    test_augment,
                                    add_feat_axis=args.add_feat_axis,
                                    output_size=inshape,
                                    max_angle=args.max_angle,
                                    rotate_prob=args.rotate_prob,) 

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers)

        print("Test on test set", args.load_model)

        csv_name = model_dir + "/intra-subject_test.csv"
        if os.path.exists(csv_name):
            os.remove(csv_name)

        utils.validate_pair(csv_name, test_loader, model, 
                  model_dir, 
                  device,
                  inshape,
                  logger)

    ############################################################################

    # test on the training set
    if args.train_pairs is not None:

        print("=> Test on single image pair for Train Set")

        # prepare model directory
        model_dir = args.ROOT + "/test_train"
        try:
            shutil.rmtree(model_dir)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))

        os.makedirs(model_dir, exist_ok=True)

        # create logger
        logger = get_logger('RDA test for model' + model_dir, log_file=model_dir + '/test.log')
        logger.info(f'Start test on {device}.')

        train_dataset = generators.DatasetRISI3DPair(
                                    args.train_pairs,
                                    train_augment,
                                    add_feat_axis=args.add_feat_axis,
                                    output_size=inshape,
                                    max_angle=args.max_angle,
                                    rotate_prob=args.rotate_prob,)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            # sampler = train_sampler,
            num_workers=args.workers)
            # persistent_workers=True)

        print("Test on training set", args.load_model)

        csv_name = model_dir + "/intra-subject_train.csv"
        if os.path.exists(csv_name):
            os.remove(csv_name)

        utils.validate_pair(csv_name, train_loader, model, 
                  model_dir, 
                  device,
                  inshape,
                  logger)

    ############################################################################

    # test on the evaluation set
    if args.eval_pairs is not None:

        print("=> Test on single image pair for Evaluation Set")

        # prepare model directory
        model_dir = args.ROOT + "/test_eval"
        try:
            shutil.rmtree(model_dir)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))

        os.makedirs(model_dir, exist_ok=True)

        # create logger
        logger = get_logger('RDA test for model' + model_dir, log_file=model_dir + '/test.log')
        logger.info(f'Start test on {device}.')

        eval_dataset = generators.DatasetRISI3DPair(
                                    args.eval_pairs,
                                    eval_augment,
                                    add_feat_axis=args.add_feat_axis,
                                    output_size=inshape,
                                    max_angle=args.max_angle,
                                    rotate_prob=args.rotate_prob,)
        
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers)
        
        print("Test on evaluation set", args.load_model)

        csv_name = model_dir + "/intra-subject_eval.csv"
        if os.path.exists(csv_name):
            os.remove(csv_name)

        utils.validate_pair(csv_name, eval_loader, model, 
                  model_dir, 
                  device,
                  inshape,
                  logger)
    


class RDATestLauncher:

    def __init__(self, parse):
        ############################# data organization parameters ############################
        parse.add_argument('--gpu', default='0', 
                           help='GPU ID number(s), comma-separated (default: 0)')
        parse.add_argument('--train-double-pairs', default=None, 
                            help='line-seperated list of training files')
        parse.add_argument('--eval-double-pairs', default=None, 
                            help='line-seperated list of validation files')
        parse.add_argument('--train-pairs', default=None, 
                            help='line-seperated list of test files')
        parse.add_argument('--eval-pairs', default=None, 
                            help='line-seperated list of test files')
        parse.add_argument('--test-pairs', default=None, 
                            help='line-seperated list of test files')
        parse.add_argument('--img-prefix', help='optional input image file prefix')
        parse.add_argument('--img-suffix', help='optional input image file suffix')
        parse.add_argument('--ROOT', metavar='DIR',
                            default="/data/mengjin/RegionalDeepAtrophy",
                            help='directory to save models and logs')
        parse.add_argument('--add-feat-axis', default=False,
                            help='when loading with gpu, do not need to add feature axis.')
        parse.add_argument('--save-image', default=True,
                            help='Whether to save tested images to .nii.gz file formats')
        parse.add_argument('--save-image-freq', default=10,
                            help='Whether to save tested images to .nii.gz file formats')
        parse.add_argument('-j', '--workers', default=0, type=int, metavar='N', # 12
                            help='number of data loading workers (default: 0)')

        ############################# training parameters ############################
        parse.add_argument('--n-classes', type=int, default=2, 
                            help='number of output classes for unet segmentation')
                            # default = 2: [nothing, shrink, expand]
        parse.add_argument('--in-dim', type=int, default=2, 
                            help='default dimension for unet seg input')
                            # 2: input 2 images for [bl, fu] for two attention maps
        parse.add_argument('--out-dim', type=int, default=3, 
                            help='default dimension for unet seg output')
                            # 1: output attention map directly,
                            # 2: output 2 channels mimicking segmentation images
                            # 3: output 3 channels of [shrink, bg, expand] voxels
        parse.add_argument('--input-D',default=48,type=int,
                           help='Input size of depth')
        parse.add_argument('--input-H', default=80, type=int,
                           help='Input size of height')
        parse.add_argument('--input-W', default=64, type=int,
                           help='Input size of width')
        parse.add_argument('--hyper-a-sigmoid', type=float, default=0.02, 
                            help='default hyper parameter for sigmoid for RISI loss')
        parse.add_argument('--risi-loss-weight', type=float, default=1, 
                            help='default weight of risi loss')
        parse.add_argument('--load-model',
                            default="",
                            # default="/voxelmorph_20210907/Model/2023-08-20_20-30/best_checkpoint.pt", # ncc = 5.33, /last_checkpoint_0030.pt
                            help='optional model file to initialize with')
        parse.add_argument('--load-unet',
                            default="",
                            help='optional model file to initialize with')
        parse.add_argument('--batch-size', type=int, default=128, 
                            help='barch size')
        parse.add_argument('--max-angle', type=int, default=45,
                            help='maximum rotation angle (default: 45)')
        parse.add_argument('--rotate-prob', type=float, default=0.5,
                            help='probability of rotation (default: 0.5)')

        # Set the function to run
        # parse.set_defaults(func = lambda args : self.run(args))
        parse.set_defaults(func = lambda args : self.run(args))

    def run(self, args):

        main(args)