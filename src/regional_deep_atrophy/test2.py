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

def main(args):

    print("ROOT is ", args.ROOT)

    if not os.path.exists(args.ROOT + '/Model'):
        os.makedirs(args.ROOT + '/Model')

    if not os.path.exists(args.ROOT + '/log'):
        os.makedirs(args.ROOT + '/log')

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

    # prepare model directory
    model_dir = args.ROOT + "/test"
    try:
        shutil.rmtree(model_dir)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

    os.makedirs(model_dir, exist_ok=True)


    ############################################################################

    # test on the training set
    if args.train_pairs is not None:

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
            shuffle=True,
            # sampler = train_sampler,
            num_workers=args.workers)
            # persistent_workers=True)

        print("Test on training set", args.load_model)

        # test
        epoch_step_time = []

        csv_name = model_dir + "/intra-subject_train.csv"
        if os.path.exists(csv_name):
            os.remove(csv_name)

    ############################################################################

    # test on the evaluation set
    if args.eval_pairs is not None:
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
            shuffle=True,
            num_workers=args.workers)
        
        print("Test on evaluation set", args.load_model)

        # test
        epoch_step_time = []

        csv_name = model_dir + "/intra-subject_eval.csv"
        if os.path.exists(csv_name):
            os.remove(csv_name)
    
    ############################################################################

    # test on the test set
    if args.test_pairs is not None:

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
            shuffle=True,
            num_workers=args.workers)

        print("Test on test set", args.load_model)

        # test
        epoch_step_time = []

        csv_name = model_dir + "/intra-subject_test.csv"
        if os.path.exists(csv_name):
            os.remove(csv_name)

    with torch.no_grad():
        with open(csv_name, 'a', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(["moving_fname", "fixed_fname",
                            "subjectID", "side", "stage", "bl_time", "fu_time",
                            "date_diff", "label_date_diff",
                            "date_diff_pred1", "date_diff_pred2",
                            "date_diff_acc1", "date_diff_acc2",
                            # "vol_diff_ratio_hippo", "vol_diff_ratio_sulcus", 
                            "bl_hippo_volume", "fu_hippo_volume",
                            "bl_sulcus_volume", "fu_sulcus_volume",
                            "hippo_attn_voxel", "sulcus_attn_voxel",])
    

            for i, sample in enumerate(test_loader):
                print(i, "new batch!",  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                step_start_time = time.time()

                # generate inputs (and true outputs) and convert them to tensors
                sample["imgs_seg"] = sample["imgs_seg"].squeeze(1).to(device).float() # .permute(0, 4, 1, 2, 3)
                sample["imgs_bl"] = sample["imgs_bl"].squeeze(1).to(device).float() 
                sample["imgs_fu"] = sample["imgs_fu"].squeeze(1).to(device).float() 
                sample["imgs_warp"] = sample["imgs_warp"].squeeze(1).to(device).float() 

                sample["label_date_diff"] = sample["label_date_diff"].long()

                x = torch.cat([sample["imgs_bl"], sample["imgs_fu"] ], dim=1)
                out_seg = model.UNet3D_Seg(x)
                out_seg, moved_img, volume_diff, volume, warped_volume, jdet = model.attention(sample["imgs_bl"],
                                sample["imgs_fu"],
                                sample["imgs_warp"],
                                out_seg,
                                registration=True)

                volume_diff_divide_hippo, volume_diff_divide_sulcus = volume_diff
                hippo_volume, sulcus_volume = volume
                warped_hippo_volume, warped_sulcus_volume = warped_volume

                volume_diff_divide_hippo = 50 * torch.log(volume_diff_divide_hippo)
                volume_diff_divide_hippo = torch.stack((volume_diff_divide_hippo, - volume_diff_divide_hippo), dim=1)

                volume_diff_divide_sulcus = 50 * torch.log(volume_diff_divide_sulcus)
                volume_diff_divide_sulcus = torch.stack((- volume_diff_divide_sulcus, volume_diff_divide_sulcus), dim=1)

                vol_diff_label = [volume_diff_divide_hippo, volume_diff_divide_sulcus]
                attention_pred = torch.argmax(out_seg, dim=1)

                # calculate this number by attention sum of voxels
                hippo_attn_vox = (attention_pred == 1).sum(dim=(1, 2, 3))
                sulcus_attn_vox = (attention_pred == 2).sum(dim=(1, 2, 3))

                date_diff_pred1 = torch.argmax(vol_diff_label[0], dim=1)
                date_diff_pred2 = torch.argmax(vol_diff_label[1], dim=1)
                
                for i in range(len(sample["label_date_diff"])):

                    moving_name = sample["bl_fname"][i].split("/")[-1]
                    print("moving_name", moving_name)
                    moving_seg_name = model_dir + "/" + moving_name.replace('bl_to_hw', "seg_to_hw", 1)

                    subjectID = moving_name[:10]
                    #TODO
                    stage = sample["stage"][i]
                    side = sample["side"][i]
                    bl_time = moving_name[11:21]
                    fu_time = moving_name[22:32]

                    moving_name = model_dir + "/" + moving_name
                    fixed_name = moving_name.replace('bl_to_hw', "fu_to_hw", 1)
                    moved_name = moving_name.replace('.nii.gz', "_moved.nii.gz", 1)
                    warp_name = moving_name.replace('.nii.gz', "_warp.nii.gz", 1)

                    # gather information of subjects
                    row = [moving_name, fixed_name,
                            subjectID, side, stage, bl_time, fu_time,
                            sample["date_diff"][i].item(), sample["label_date_diff"][i].item(),
                            date_diff_pred1[i].item(), date_diff_pred2[i].item(),
                            int(sample["label_date_diff"][i].item() == date_diff_pred1[i].item()), int(sample["label_date_diff"][i].item() == date_diff_pred2[i].item()),
                            #    vol_diff_label[0][i][0].item(), vol_diff_label[1][i][0].item(),
                            hippo_volume[i].item(), warped_hippo_volume[i].item(),
                            sulcus_volume[i].item(), warped_sulcus_volume[i].item(),
                            hippo_attn_vox[i].item(), sulcus_attn_vox[i].item(),
                            ]
                    
                    # print(i)
                    # print("vol_diff_label                                                 ", 
                    #       vol_diff_label[0][i][0].item(), vol_diff_label[1][i][0].item())
                    # print("hippo_volume.item(), warped_hippo_volume.item()                ", 
                    #       hippo_volume[i].item(), warped_hippo_volume[i].item())
                    # print("sulcus_volume.item(), warped_sulcus_volume.item()              ", 
                    #       sulcus_volume[i].item(), warped_sulcus_volume[i].item())
                    # print("hippo_attn_vox.item(), sulcus_attn_vox.item()                  ",
                    #       hippo_attn_vox[i].item(), sulcus_attn_vox[i].item())

                    wr.writerow(row)
                    # print(row)

                    # save example images, sample the last images of each batch
                    affine_trans = [sample["affines"][i], 
                                    [dim[i].item() for dim in sample["orig_shapes"]],
                                    [dim[i].item() for dim in sample["crop_posns"]]]

                    # save bl image
                    shutil.copyfile(sample["bl_fname"][i], moving_name) 

                    # save attention image
                    moving_attention_name = moving_name.replace('.nii.gz', "_attention.nii.gz", 1)
                    utils.save_volfile(attention_pred[i, :].detach().cpu().numpy().squeeze().astype(float),
                                       moving_attention_name, affine_trans)

                    # save position mask
                    moving_pos_mask = moving_name.replace('.nii.gz', "_position_mask.nii.gz", 1)
                    utils.save_volfile(np.ones(inshape), moving_pos_mask, affine_trans)

                    # save jacobian determinant
                    moving_jdet_name = moving_name.replace('.nii.gz', "_jdet.nii.gz", 1)
                    utils.save_volfile(jdet[i, 0, :].detach().cpu().numpy().squeeze(),
                                        moving_jdet_name, affine_trans)

                    # # save seg image before registration
                    utils.save_volfile(sample["imgs_seg"][i].detach().cpu().numpy().squeeze(), moving_seg_name, affine_trans)

                    # save moved image
                    utils.save_volfile(moved_img[i, 0, :].detach().cpu().numpy().squeeze(),
                                                moved_name, affine_trans)

                    # get compute time
                    epoch_step_time.append(time.time() - step_start_time)

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